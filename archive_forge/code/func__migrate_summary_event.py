from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.audio import metadata as audio_metadata
from tensorboard.plugins.custom_scalar import (
from tensorboard.plugins.graph import metadata as graphs_metadata
from tensorboard.plugins.histogram import metadata as histograms_metadata
from tensorboard.plugins.hparams import metadata as hparams_metadata
from tensorboard.plugins.image import metadata as images_metadata
from tensorboard.plugins.mesh import metadata as mesh_metadata
from tensorboard.plugins.pr_curve import metadata as pr_curves_metadata
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.plugins.text import metadata as text_metadata
from tensorboard.util import tensor_util
def _migrate_summary_event(event, initial_metadata):
    values = event.summary.value
    new_values = [new for old in values for new in _migrate_value(old, initial_metadata)]
    if len(values) == len(new_values) and all((x is y for x, y in zip(values, new_values))):
        return (event,)
    del event.summary.value[:]
    event.summary.value.extend(new_values)
    return (event,)