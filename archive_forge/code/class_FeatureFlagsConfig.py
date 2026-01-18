from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import hashlib
import logging
import os
import threading
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files as file_utils
class FeatureFlagsConfig:
    """Stores all Property Objects for a given FeatureFlagsConfig."""

    def __init__(self, feature_flags_config_yaml, account_id=None, project_id=None):
        self.user_key = account_id or config.GetCID()
        self.project_id = project_id
        self.properties = _ParseFeatureFlagsConfig(feature_flags_config_yaml)

    def Get(self, prop):
        """Returns the value for the given property."""
        prop_str = str(prop)
        if prop_str not in self.properties:
            return None
        total_weight = sum(self.properties[prop_str].weights)
        if self.project_id:
            hash_str = prop_str + self.project_id
        else:
            hash_str = prop_str + self.user_key
        project_hash = int(hashlib.sha256(hash_str.encode('utf-8')).hexdigest(), 16) % total_weight
        list_of_weights = self.properties[prop_str].weights
        sum_of_weights = 0
        for i in range(len(list_of_weights)):
            sum_of_weights += list_of_weights[i]
            if project_hash < sum_of_weights:
                return self.properties[prop_str].values[i]