import io
import json
import yaml
from heatclient.common import format_utils
from heatclient.tests.unit.osc import utils
class ShowValue(format_utils.ValueFormat):

    def take_action(self, parsed_args):
        return (columns, data)