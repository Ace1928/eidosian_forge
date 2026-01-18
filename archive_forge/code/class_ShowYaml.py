import io
import json
import yaml
from heatclient.common import format_utils
from heatclient.tests.unit.osc import utils
class ShowYaml(format_utils.YamlFormat):

    def take_action(self, parsed_args):
        return (columns, data)