import functools
import importlib.metadata
import importlib.util
import json
import numbers
import os
import pickle
import shutil
import sys
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union
import numpy as np
from .. import __version__ as version
from ..utils import flatten_dict, is_datasets_available, is_pandas_available, is_torch_available, logging
from ..trainer_callback import ProgressCallback, TrainerCallback  # noqa: E402
from ..trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, IntervalStrategy  # noqa: E402
from ..training_args import ParallelMode  # noqa: E402
from ..utils import ENV_VARS_TRUE_VALUES, is_torch_tpu_available  # noqa: E402
class FlyteCallback(TrainerCallback):
    """A [`TrainerCallback`] that sends the logs to [Flyte](https://flyte.org/).
    NOTE: This callback only works within a Flyte task.

    Args:
        save_log_history (`bool`, *optional*, defaults to `True`):
            When set to True, the training logs are saved as a Flyte Deck.

        sync_checkpoints (`bool`, *optional*, defaults to `True`):
            When set to True, checkpoints are synced with Flyte and can be used to resume training in the case of an
            interruption.

    Example:

    ```python
    # Note: This example skips over some setup steps for brevity.
    from flytekit import current_context, task


    @task
    def train_hf_transformer():
        cp = current_context().checkpoint
        trainer = Trainer(..., callbacks=[FlyteCallback()])
        output = trainer.train(resume_from_checkpoint=cp.restore())
    ```
    """

    def __init__(self, save_log_history: bool=True, sync_checkpoints: bool=True):
        super().__init__()
        if not is_flytekit_available():
            raise ImportError('FlyteCallback requires flytekit to be installed. Run `pip install flytekit`.')
        if not is_flyte_deck_standard_available() or not is_pandas_available():
            logger.warning('Syncing log history requires both flytekitplugins-deck-standard and pandas to be installed. Run `pip install flytekitplugins-deck-standard pandas` to enable this feature.')
            save_log_history = False
        from flytekit import current_context
        self.cp = current_context().checkpoint
        self.save_log_history = save_log_history
        self.sync_checkpoints = sync_checkpoints

    def on_save(self, args, state, control, **kwargs):
        if self.sync_checkpoints and state.is_world_process_zero:
            ckpt_dir = f'checkpoint-{state.global_step}'
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            logger.info(f'Syncing checkpoint in {ckpt_dir} to Flyte. This may take time.')
            self.cp.save(artifact_path)

    def on_train_end(self, args, state, control, **kwargs):
        if self.save_log_history:
            import pandas as pd
            from flytekit import Deck
            from flytekitplugins.deck.renderer import TableRenderer
            log_history_df = pd.DataFrame(state.log_history)
            Deck('Log History', TableRenderer().to_html(log_history_df))