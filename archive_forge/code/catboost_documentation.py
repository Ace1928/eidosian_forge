from pathlib import Path
from types import SimpleNamespace
from typing import List, Union
from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry
`log_summary` logs useful metrics about catboost model after training is done.

    Arguments:
        model: it can be CatBoostClassifier or CatBoostRegressor.
        log_all_params: (boolean) if True (default) log the model hyperparameters as W&B config.
        save_model_checkpoint: (boolean) if True saves the model upload as W&B artifacts.
        log_feature_importance: (boolean) if True (default) logs feature importance as W&B bar chart using the default setting of `get_feature_importance`.

    Using this along with `wandb_callback` will:

    - save the hyperparameters as W&B config,
    - log `best_iteration` and `best_score` as `wandb.summary`,
    - save and upload your trained model to Weights & Biases Artifacts (when `save_model_checkpoint = True`)
    - log feature importance plot.

    Example:
        ```python
        train_pool = Pool(train[features], label=train["label"], cat_features=cat_features)
        test_pool = Pool(test[features], label=test["label"], cat_features=cat_features)

        model = CatBoostRegressor(
            iterations=100,
            loss_function="Cox",
            eval_metric="Cox",
        )

        model.fit(
            train_pool,
            eval_set=test_pool,
            callbacks=[WandbCallback()],
        )

        log_summary(model)
        ```
    