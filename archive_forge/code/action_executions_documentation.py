import logging
from oslo_serialization import jsonutils
from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient import utils
Delete action execution.