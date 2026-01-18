import inspect
import functools
from enum import Enum
import torch.autograd
def _check_iterator_valid(datapipe, iterator_id, next_method_exists=False) -> None:
    """
    Given an instance of a DataPipe and an iterator ID, check if the IDs match, and if not, raises an exception.

    In the case of ChildDataPipe, the ID gets compared to the one stored in `main_datapipe` as well.
    """
    if next_method_exists:
        if datapipe._valid_iterator_id is not None and datapipe._valid_iterator_id != 0:
            extra_msg = "\nNote that this exception is raised inside your IterDataPipe's a `__next__` method"
            raise RuntimeError(_gen_invalid_iterdatapipe_msg(datapipe) + extra_msg + _feedback_msg)
    elif hasattr(datapipe, '_is_child_datapipe') and datapipe._is_child_datapipe is True:
        if hasattr(datapipe, '_check_valid_iterator_id'):
            if not datapipe._check_valid_iterator_id(iterator_id):
                raise RuntimeError(f'This iterator has been invalidated, because a new iterator has been created from one of the ChildDataPipes of {_generate_iterdatapipe_msg(datapipe.main_datapipe)}.' + _feedback_msg)
        else:
            raise RuntimeError('ChildDataPipe must have method `_check_valid_iterator_id`.')
    elif datapipe._valid_iterator_id != iterator_id:
        raise RuntimeError(_gen_invalid_iterdatapipe_msg(datapipe) + _feedback_msg)