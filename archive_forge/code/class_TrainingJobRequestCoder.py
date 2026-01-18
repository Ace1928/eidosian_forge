import csv
import datetime
import json
import logging
import apache_beam as beam
from six.moves import cStringIO
import yaml
from google.cloud.ml.util import _decoders
from google.cloud.ml.util import _file
class TrainingJobRequestCoder(beam.coders.Coder):
    """Custom coder for a TrainingJobRequest object."""

    def encode(self, training_job_request):
        """Encode a TrainingJobRequest to a JSON string.

    Args:
      training_job_request: A TrainingJobRequest object.

    Returns:
      A JSON string
    """
        d = {}
        d.update(training_job_request.__dict__)
        for k in ['timeout', 'polling_interval']:
            if d[k]:
                d[k] = d[k].total_seconds()
        return json.dumps(d)

    def decode(self, training_job_request_string):
        """Decode a JSON string representing a TrainingJobRequest.

    Args:
      training_job_request_string: A string representing a TrainingJobRequest.

    Returns:
      TrainingJobRequest object.
    """
        r = TrainingJobRequest()
        d = json.loads(training_job_request_string)
        for k in ['timeout', 'polling_interval']:
            if d[k]:
                d[k] = datetime.timedelta(seconds=d[k])
        r.__dict__.update(d)
        return r