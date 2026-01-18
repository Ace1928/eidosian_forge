from __future__ import absolute_import
import argparse
import json
import sys
import ruamel.yaml as yaml
from googlecloudsdk.third_party.appengine.admin.tools.conversion import yaml_schema_v1
from googlecloudsdk.third_party.appengine.admin.tools.conversion import yaml_schema_v1beta
A script for converting between legacy YAML and public JSON representation.

Example invocation:
  convert_yaml.py app.yaml > app.json
