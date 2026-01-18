import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import wandb
class ValidationDataLogger:
    """Logs validation data as a wandb.Table.

    ValidationDataLogger is intended to be used inside of library integrations
    in order to facilitate the process of optionally building a validation dataset
    and logging periodic predictions against such validation data using WandB best
    practices.
    """
    validation_inputs: Union[Sequence, Dict[str, Sequence]]
    validation_targets: Optional[Union[Sequence, Dict[str, Sequence]]]
    validation_indexes: List['_TableIndex']
    prediction_row_processor: Optional[Callable]
    class_labels_table: Optional['wandb.Table']
    infer_missing_processors: bool

    def __init__(self, inputs: Union[Sequence, Dict[str, Sequence]], targets: Optional[Union[Sequence, Dict[str, Sequence]]]=None, indexes: Optional[List['_TableIndex']]=None, validation_row_processor: Optional[Callable]=None, prediction_row_processor: Optional[Callable]=None, input_col_name: str='input', target_col_name: str='target', table_name: str='wb_validation_data', artifact_type: str='validation_dataset', class_labels: Optional[List[str]]=None, infer_missing_processors: bool=True) -> None:
        """Initialize a new ValidationDataLogger.

        Args:
            inputs: A list of input vectors or dictionary of lists of input vectors
                (used if the model has multiple named inputs)
            targets: A list of target vectors or dictionary of lists of target vectors
                (used if the model has multiple named targets/putputs). Defaults to `None`.
                `targets` and `indexes` cannot both be `None`.
            indexes: An ordered list of `wandb.data_types._TableIndex` mapping the
                input items to their source table. This is most commonly retrieved by using
                `indexes = my_data_table.get_index()`. Defaults to `None`. `targets`
                and `indexes` cannot both be `None`.
            validation_row_processor: A function to apply to the validation data,
                commonly used to visualize the data. The function will receive an `ndx` (`int`)
                and a `row` (`dict`). If `inputs` is a list, then `row["input"]` will be the input
                data for the row. Else, it will be keyed based on the name of the input slot
                (corresponding to `inputs`). If `targets` is a list, then
                `row["target"]` will be the target data for the row. Else, it will
                be keyed based on `targets`. For example, if your input data is a
                single ndarray, but you wish to visualize the data as an image,
                then you can provide `lambda ndx, row: {"img": wandb.Image(row["input"])}`
                as the processor. If `None`, we will try to guess the appropriate processor.
                Ignored if `log_evaluation` is `False` or `val_keys` are present. Defaults to `None`.
            prediction_row_processor: Same as validation_row_processor, but applied to the
                model's output. `row["output"]` will contain the results of the model output.
                Defaults to `None`.
            input_col_name: The name to use for the input column.
                Defaults to `"input"`.
            target_col_name: The name to use for the target column.
                Defaults to `"target"`.
            table_name: The name to use for the validation table.
                Defaults to `"wb_validation_data"`.
            artifact_type: The artifact type to use for the validation data.
                Defaults to `"validation_dataset"`.
            class_labels: Optional list of lables to use in the inferred
                processors. If the model's `target` or `output` is inferred to be a class,
                we will attempt to map the class to these labels. Defaults to `None`.
            infer_missing_processors: Determines if processors are inferred if
                they are missing. Defaults to True.
        """
        class_labels_table: Optional[wandb.Table]
        if isinstance(class_labels, list) and len(class_labels) > 0:
            class_labels_table = wandb.Table(columns=['label'], data=[[label] for label in class_labels])
        else:
            class_labels_table = None
        if indexes is None:
            assert targets is not None
            local_validation_table = wandb.Table(columns=[], data=[])
            if isinstance(targets, dict):
                for col_name in targets:
                    local_validation_table.add_column(col_name, targets[col_name])
            else:
                local_validation_table.add_column(target_col_name, targets)
            if isinstance(inputs, dict):
                for col_name in inputs:
                    local_validation_table.add_column(col_name, inputs[col_name])
            else:
                local_validation_table.add_column(input_col_name, inputs)
            if validation_row_processor is None and infer_missing_processors:
                example_input = _make_example(inputs)
                example_target = _make_example(targets)
                if example_input is not None and example_target is not None:
                    validation_row_processor = _infer_validation_row_processor(example_input, example_target, class_labels_table, input_col_name, target_col_name)
            if validation_row_processor is not None:
                local_validation_table.add_computed_columns(validation_row_processor)
            local_validation_artifact = wandb.Artifact(table_name, artifact_type)
            local_validation_artifact.add(local_validation_table, 'validation_data')
            if wandb.run:
                wandb.run.use_artifact(local_validation_artifact)
            indexes = local_validation_table.get_index()
        else:
            local_validation_artifact = None
        self.class_labels_table = class_labels_table
        self.validation_inputs = inputs
        self.validation_targets = targets
        self.validation_indexes = indexes
        self.prediction_row_processor = prediction_row_processor
        self.infer_missing_processors = infer_missing_processors
        self.local_validation_artifact = local_validation_artifact
        self.input_col_name = input_col_name

    def make_predictions(self, predict_fn: Callable) -> Union[Sequence, Dict[str, Sequence]]:
        """Produce predictions by passing `validation_inputs` to `predict_fn`.

        Args:
            predict_fn (Callable): Any function which can accept `validation_inputs` and produce
                a list of vectors or dictionary of lists of vectors

        Returns:
            (Sequence | Dict[str, Sequence]): The returned value of predict_fn
        """
        return predict_fn(self.validation_inputs)

    def log_predictions(self, predictions: Union[Sequence, Dict[str, Sequence]], prediction_col_name: str='output', val_ndx_col_name: str='val_row', table_name: str='validation_predictions', commit: bool=True) -> wandb.data_types.Table:
        """Log a set of predictions.

        Intended usage:

        vl.log_predictions(vl.make_predictions(self.model.predict))

        Args:
            predictions (Sequence | Dict[str, Sequence]): A list of prediction vectors or dictionary
                of lists of prediction vectors
            prediction_col_name (str, optional): the name of the prediction column. Defaults to "output".
            val_ndx_col_name (str, optional): The name of the column linking prediction table
                to the validation ata table. Defaults to "val_row".
            table_name (str, optional): name of the prediction table. Defaults to "validation_predictions".
            commit (bool, optional): determines if commit should be called on the logged data. Defaults to False.
        """
        pred_table = wandb.Table(columns=[], data=[])
        if isinstance(predictions, dict):
            for col_name in predictions:
                pred_table.add_column(col_name, predictions[col_name])
        else:
            pred_table.add_column(prediction_col_name, predictions)
        pred_table.add_column(val_ndx_col_name, self.validation_indexes)
        if self.prediction_row_processor is None and self.infer_missing_processors:
            example_prediction = _make_example(predictions)
            example_input = _make_example(self.validation_inputs)
            if example_prediction is not None and example_input is not None:
                self.prediction_row_processor = _infer_prediction_row_processor(example_prediction, example_input, self.class_labels_table, self.input_col_name, prediction_col_name)
        if self.prediction_row_processor is not None:
            pred_table.add_computed_columns(self.prediction_row_processor)
        wandb.log({table_name: pred_table}, commit=commit)
        return pred_table