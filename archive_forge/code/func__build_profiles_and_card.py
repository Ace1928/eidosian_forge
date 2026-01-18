import importlib
import logging
import os
import sys
import time
import cloudpickle
from packaging.version import Version
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.recipes.artifacts import DataframeArtifact, TransformerArtifact
from mlflow.recipes.cards import BaseCard
from mlflow.recipes.step import BaseStep, StepClass
from mlflow.recipes.utils.execution import get_step_output_path
from mlflow.recipes.utils.step import get_pandas_data_profiles, validate_classification_config
from mlflow.recipes.utils.tracking import TrackingConfig, get_recipe_tracking_config
def _build_profiles_and_card(self, train_df, train_transformed, transformer) -> BaseCard:
    card = BaseCard(self.recipe_name, self.name)
    if not self.skip_data_profiling:
        train_transformed_profile = get_pandas_data_profiles([['Profile of Train Transformed Dataset', train_transformed]])
        card.add_tab('Data Profile (Train Transformed)', '{{PROFILE}}').add_pandas_profile('PROFILE', train_transformed_profile)
    from sklearn import set_config
    from sklearn.utils import estimator_html_repr
    set_config(display='diagram')
    transformer_repr = estimator_html_repr(transformer)
    card.add_tab('Transformer', '{{TRANSFORMER}}').add_html('TRANSFORMER', transformer_repr)
    card.add_tab('Input Schema', '{{INPUT_SCHEMA}}').add_html('INPUT_SCHEMA', BaseCard.render_table(({'Name': n, 'Type': t} for n, t in train_df.dtypes.items())))
    try:
        card.add_tab('Output Schema', '{{OUTPUT_SCHEMA}}').add_html('OUTPUT_SCHEMA', BaseCard.render_table(({'Name': n, 'Type': t} for n, t in train_transformed.dtypes.items())))
    except Exception as e:
        card.add_tab('Output Schema', '{{OUTPUT_SCHEMA}}').add_html('OUTPUT_SCHEMA', f'Failed to extract transformer schema. Error: {e}')
    card.add_tab('Data Preview', '{{DATA_PREVIEW}}').add_html('DATA_PREVIEW', BaseCard.render_table(train_transformed.head()))
    card.add_tab('Run Summary', '\n                {{ EXE_DURATION }}\n                {{ LAST_UPDATE_TIME }}\n                ')
    return card