from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def ParseJobName(name):
    """Parses the id from a full hyperparameter tuning job name."""
    return resources.REGISTRY.Parse(name, collection=HPTUNING_JOB_COLLECTION).Name()