import abc
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.recipes.artifacts import DataframeArtifact
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep, StepClass
from mlflow.recipes.steps.ingest.datasets import (
from mlflow.recipes.utils.step import get_pandas_data_profiles, validate_classification_config
from mlflow.utils.file_utils import read_parquet_as_pandas_df
class BaseIngestStep(BaseStep, metaclass=abc.ABCMeta):
    _DATASET_FORMAT_SPARK_TABLE = 'spark_table'
    _DATASET_FORMAT_DELTA = 'delta'
    _DATASET_FORMAT_PARQUET = 'parquet'
    _DATASET_PROFILE_OUTPUT_NAME = 'dataset_profile.html'
    _STEP_CARD_OUTPUT_NAME = 'card.pkl'
    _SUPPORTED_DATASETS = [ParquetDataset, DeltaTableDataset, SparkSqlDataset, CustomDataset]

    def _validate_and_apply_step_config(self):
        dataset_format = self.step_config.get('using')
        if not dataset_format:
            raise MlflowException(message='Dataset format must be specified via the `using` key within the `ingest` section of recipe.yaml', error_code=INVALID_PARAMETER_VALUE)
        if self.step_class() == StepClass.TRAINING:
            self.target_col = self.step_config.get('target_col')
            if self.target_col is None:
                raise MlflowException('Missing target_col config in recipe config.', error_code=INVALID_PARAMETER_VALUE)
            self.positive_class = self.step_config.get('positive_class')
        for dataset_class in BaseIngestStep._SUPPORTED_DATASETS:
            if dataset_class.handles_format(dataset_format):
                self.dataset = dataset_class.from_config(dataset_config=self.step_config, recipe_root=self.recipe_root)
                break
        else:
            raise MlflowException(message=f'Unrecognized dataset format: {dataset_format}', error_code=INVALID_PARAMETER_VALUE)
        self.skip_data_profiling = self.step_config.get('skip_data_profiling', False)

    def _run(self, output_directory: str) -> BaseCard:
        dataset_dst_path = os.path.abspath(os.path.join(output_directory, self.dataset_output_name))
        self.dataset.resolve_to_parquet(dst_path=dataset_dst_path)
        _logger.debug("Successfully stored data in parquet format at '%s'", dataset_dst_path)
        ingested_df = read_parquet_as_pandas_df(data_parquet_path=dataset_dst_path)
        if self.step_class() == StepClass.TRAINING:
            if self.target_col not in ingested_df.columns:
                raise MlflowException(f"Target column '{self.target_col}' not found in ingested dataset.", error_code=INVALID_PARAMETER_VALUE)
            if self.task == 'classification':
                validate_classification_config(self.task, self.positive_class, ingested_df, self.target_col)
                cardinality = ingested_df[self.target_col].nunique()
                if cardinality > 2 and self.positive_class is not None:
                    raise MlflowException(f"Target column '{self.target_col}' must have a cardinality of 2,found '{cardinality}'.", error_code=INVALID_PARAMETER_VALUE)
                if self.positive_class is not None and cardinality != 2:
                    raise MlflowException(f"`For binary classification problems, target column '{self.target_col}' must have a cardinality of 2,found '{cardinality}'. `positive_class` was set, so we treat this problem as a binary classification problem. ", error_code=INVALID_PARAMETER_VALUE)
        ingested_dataset_profile = None
        if not self.skip_data_profiling:
            _logger.debug('Profiling ingested dataset')
            ingested_dataset_profile = get_pandas_data_profiles([['Profile of Ingested Dataset', ingested_df]])
            dataset_profile_path = Path(str(os.path.join(output_directory, BaseIngestStep._DATASET_PROFILE_OUTPUT_NAME)))
            dataset_profile_path.write_text(ingested_dataset_profile, encoding='utf-8')
            _logger.debug(f"Wrote dataset profile to '{dataset_profile_path}'")
        schema = pd.io.json.build_table_schema(ingested_df, index=False)
        return self._build_step_card(ingested_dataset_profile=ingested_dataset_profile, ingested_rows=len(ingested_df), schema=schema, data_preview=ingested_df.head(), dataset_src_location=getattr(self.dataset, 'location', None), dataset_sql=getattr(self.dataset, 'sql', None))

    def _build_step_card(self, ingested_dataset_profile: str, ingested_rows: int, schema: Dict, data_preview: pd.DataFrame=None, dataset_src_location: Optional[str]=None, dataset_sql: Optional[str]=None) -> BaseCard:
        """
        Constructs a step card instance corresponding to the current ingest step state.

        Args:
            ingested_dataset_path: The local filesystem path to the ingested parquet dataset file.
            dataset_src_location: The source location of the dataset (e.g. '/tmp/myfile.parquet',
                's3://mybucket/mypath', ...), if the dataset is a location-based dataset. Either
                ``dataset_src_location`` or ``dataset_sql`` must be specified.
            dataset_sql: The Spark SQL query string that defines the dataset
                (e.g. 'SELECT * FROM my_spark_table'), if the dataset is a Spark SQL dataset. Either
                ``dataset_src_location`` or ``dataset_sql`` must be specified.

        Returns:
            An BaseCard instance corresponding to the current ingest step state.

        """
        if dataset_src_location is None and dataset_sql is None:
            raise MlflowException(message='Failed to build step card because neither a dataset location nor a dataset Spark SQL query were specified', error_code=INVALID_PARAMETER_VALUE)
        card = BaseCard(self.recipe_name, self.name)
        if not self.skip_data_profiling:
            card.add_tab('Data Profile', '{{PROFILE}}').add_pandas_profile('PROFILE', ingested_dataset_profile)
        schema_html = BaseCard.render_table(schema['fields'])
        card.add_tab('Data Schema', '{{SCHEMA}}').add_html('SCHEMA', schema_html)
        if data_preview is not None:
            card.add_tab('Data Preview', '{{DATA_PREVIEW}}').add_html('DATA_PREVIEW', BaseCard.render_table(data_preview))
        card.add_tab('Run Summary', '{{ INGESTED_ROWS }}' + '{{ DATA_SOURCE }}' + '{{ EXE_DURATION }}' + '{{ LAST_UPDATE_TIME }}').add_markdown(name='INGESTED_ROWS', markdown=f'**Number of rows ingested:** `{ingested_rows}`').add_markdown(name='DATA_SOURCE', markdown=f'**Dataset source location:** `{dataset_src_location}`' if dataset_src_location is not None else f'**Dataset SQL:** `{dataset_sql}`')
        return card