import abc
import collections
import json
from tensorboard.uploader import util
class ReadableFormatter(BaseExperimentFormatter):
    """A formatter implementation that outputs human-readable text."""
    _NAME_COLUMN_WIDTH = 20

    def __init__(self):
        super().__init__()

    def format_experiment(self, experiment, experiment_url):
        output = []
        output.append(experiment_url)
        data = [('Name', experiment.name or '[No Name]'), ('Description', experiment.description or '[No Description]'), ('Id', experiment.experiment_id), ('Created', util.format_time(experiment.create_time)), ('Updated', util.format_time(experiment.update_time)), ('Runs', str(experiment.num_runs)), ('Tags', str(experiment.num_tags)), ('Scalars', str(experiment.num_scalars)), ('Tensor bytes', str(experiment.total_tensor_bytes)), ('Binary object bytes', str(experiment.total_blob_bytes))]
        for name, value in data:
            output.append('\t%s %s' % (name.ljust(self._NAME_COLUMN_WIDTH), value))
        return '\n'.join(output)