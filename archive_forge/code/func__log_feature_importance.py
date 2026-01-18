from pathlib import Path
from types import SimpleNamespace
from typing import List, Union
from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
def _log_feature_importance(model: Union[CatBoostClassifier, CatBoostRegressor]) -> None:
    """Log feature importance with default settings."""
    if wandb.run is None:
        raise wandb.Error('You must call `wandb.init()` before `_checkpoint_artifact()`')
    feat_df = model.get_feature_importance(prettified=True)
    fi_data = [[feat, feat_imp] for feat, feat_imp in zip(feat_df['Feature Id'], feat_df['Importances'])]
    table = wandb.Table(data=fi_data, columns=['Feature', 'Importance'])
    wandb.log({'Feature Importance': wandb.plot.bar(table, 'Feature', 'Importance', title='Feature Importance')}, commit=False)