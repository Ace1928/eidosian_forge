import logging
from collections import defaultdict
from dataclasses import _MISSING_TYPE, dataclass, fields
from pathlib import Path
from typing import (
import pyarrow.fs
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.annotations import PublicAPI, Deprecated
from ray.widgets import Template, make_table_html_repr
from ray.data.preprocessor import Preprocessor
@dataclass
@PublicAPI(stability='stable')
class CheckpointConfig:
    """Configurable parameters for defining the checkpointing strategy.

    Default behavior is to persist all checkpoints to disk. If
    ``num_to_keep`` is set, the default retention policy is to keep the
    checkpoints with maximum timestamp, i.e. the most recent checkpoints.

    Args:
        num_to_keep: The number of checkpoints to keep
            on disk for this run. If a checkpoint is persisted to disk after
            there are already this many checkpoints, then an existing
            checkpoint will be deleted. If this is ``None`` then checkpoints
            will not be deleted. Must be >= 1.
        checkpoint_score_attribute: The attribute that will be used to
            score checkpoints to determine which checkpoints should be kept
            on disk when there are greater than ``num_to_keep`` checkpoints.
            This attribute must be a key from the checkpoint
            dictionary which has a numerical value. Per default, the last
            checkpoints will be kept.
        checkpoint_score_order: Either "max" or "min".
            If "max", then checkpoints with highest values of
            ``checkpoint_score_attribute`` will be kept.
            If "min", then checkpoints with lowest values of
            ``checkpoint_score_attribute`` will be kept.
        checkpoint_frequency: Number of iterations between checkpoints. If 0
            this will disable checkpointing.
            Please note that most trainers will still save one checkpoint at
            the end of training.
            This attribute is only supported
            by trainers that don't take in custom training loops.
        checkpoint_at_end: If True, will save a checkpoint at the end of training.
            This attribute is only supported by trainers that don't take in
            custom training loops. Defaults to True for trainers that support it
            and False for generic function trainables.
        _checkpoint_keep_all_ranks: This experimental config is deprecated.
            This behavior is now controlled by reporting `checkpoint=None`
            in the workers that shouldn't persist a checkpoint.
            For example, if you only want the rank 0 worker to persist a checkpoint
            (e.g., in standard data parallel training), then you should save and
            report a checkpoint if `ray.train.get_context().get_world_rank() == 0`
            and `None` otherwise.
        _checkpoint_upload_from_workers: This experimental config is deprecated.
            Uploading checkpoint directly from the worker is now the default behavior.
    """
    num_to_keep: Optional[int] = None
    checkpoint_score_attribute: Optional[str] = None
    checkpoint_score_order: Optional[str] = MAX
    checkpoint_frequency: Optional[int] = 0
    checkpoint_at_end: Optional[bool] = None
    _checkpoint_keep_all_ranks: Optional[bool] = _DEPRECATED_VALUE
    _checkpoint_upload_from_workers: Optional[bool] = _DEPRECATED_VALUE

    def __post_init__(self):
        if self._checkpoint_keep_all_ranks != _DEPRECATED_VALUE:
            raise DeprecationWarning("The experimental `_checkpoint_keep_all_ranks` config is deprecated. This behavior is now controlled by reporting `checkpoint=None` in the workers that shouldn't persist a checkpoint. For example, if you only want the rank 0 worker to persist a checkpoint (e.g., in standard data parallel training), then you should save and report a checkpoint if `ray.train.get_context().get_world_rank() == 0` and `None` otherwise.")
        if self._checkpoint_upload_from_workers != _DEPRECATED_VALUE:
            raise DeprecationWarning('The experimental `_checkpoint_upload_from_workers` config is deprecated. Uploading checkpoint directly from the worker is now the default behavior.')
        if self.num_to_keep is not None and self.num_to_keep <= 0:
            raise ValueError(f'Received invalid num_to_keep: {self.num_to_keep}. Must be None or an integer >= 1.')
        if self.checkpoint_score_order not in (MAX, MIN):
            raise ValueError(f'checkpoint_score_order must be either "{MAX}" or "{MIN}".')
        if self.checkpoint_frequency < 0:
            raise ValueError(f'checkpoint_frequency must be >=0, got {self.checkpoint_frequency}')

    def __repr__(self):
        return _repr_dataclass(self)

    def _repr_html_(self) -> str:
        if self.num_to_keep is None:
            num_to_keep_repr = 'All'
        else:
            num_to_keep_repr = self.num_to_keep
        if self.checkpoint_score_attribute is None:
            checkpoint_score_attribute_repr = 'Most recent'
        else:
            checkpoint_score_attribute_repr = self.checkpoint_score_attribute
        if self.checkpoint_at_end is None:
            checkpoint_at_end_repr = ''
        else:
            checkpoint_at_end_repr = self.checkpoint_at_end
        return Template('scrollableTable.html.j2').render(table=tabulate({'Setting': ['Number of checkpoints to keep', 'Checkpoint score attribute', 'Checkpoint score order', 'Checkpoint frequency', 'Checkpoint at end'], 'Value': [num_to_keep_repr, checkpoint_score_attribute_repr, self.checkpoint_score_order, self.checkpoint_frequency, checkpoint_at_end_repr]}, tablefmt='html', showindex=False, headers='keys'), max_height='none')

    @property
    def _tune_legacy_checkpoint_score_attr(self) -> Optional[str]:
        """Same as ``checkpoint_score_attr`` in ``tune.run``.

        Only used for Legacy API compatibility.
        """
        if self.checkpoint_score_attribute is None:
            return self.checkpoint_score_attribute
        prefix = ''
        if self.checkpoint_score_order == MIN:
            prefix = 'min-'
        return f'{prefix}{self.checkpoint_score_attribute}'