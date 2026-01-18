import json
import os
import io
from heatclient.common import resource_formatter
from heatclient.osc.v1 import resource
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import resources as v1_resources
class TestStackResourceListDotFormat(orchestration_fakes.TestOrchestrationv1):
    response_path = os.path.join(TEST_VAR_DIR, 'dot_test.json')
    data = 'digraph G {\n  graph [\n    fontsize=10 fontname="Verdana" compound=true rankdir=LR\n  ]\n  r_f34a35baf594b319a741 [label="rg1\nOS::Heat::ResourceGroup" ];\n  r_121e343b017a6d246f36 [label="random2\nOS::Heat::RandomString" ];\n  r_dbcae38ad41dc991751d [label="random1\nOS::Heat::RandomString" style=filled color=red];\n\n  subgraph cluster_stack_16437984473ec64a8e6c {\n    label="rg1";\n    r_30e9aa76bc0d53310cde [label="1\nOS::Heat::ResourceGroup" ];\n    r_63c05d424cb708f1599f [label="0\nOS::Heat::ResourceGroup" ];\n\n  }\n\n  subgraph cluster_stack_fbfb461c8cc84b686c08 {\n    label="1";\n    r_e2e5c36ae18e29d9c299 [label="1\nOS::Heat::RandomString" ];\n    r_56c62630a0d655bce234 [label="0\nOS::Heat::RandomString" ];\n\n  }\n\n  subgraph cluster_stack_d427657dfccc28a131a7 {\n    label="0";\n    r_240756913e2e940387ff [label="1\nOS::Heat::RandomString" ];\n    r_81c64c43d9131aceedbb [label="0\nOS::Heat::RandomString" ];\n\n  }\n\n  r_f34a35baf594b319a741 -> r_30e9aa76bc0d53310cde [\n    color=dimgray lhead=cluster_stack_16437984473ec64a8e6c arrowhead=none\n  ];\n  r_30e9aa76bc0d53310cde -> r_e2e5c36ae18e29d9c299 [\n    color=dimgray lhead=cluster_stack_fbfb461c8cc84b686c08 arrowhead=none\n  ];\n  r_63c05d424cb708f1599f -> r_240756913e2e940387ff [\n    color=dimgray lhead=cluster_stack_d427657dfccc28a131a7 arrowhead=none\n  ];\n\n  r_dbcae38ad41dc991751d -> r_121e343b017a6d246f36;\n\n}\n'

    def setUp(self):
        super(TestStackResourceListDotFormat, self).setUp()
        self.resource_client = self.app.client_manager.orchestration.resources
        self.cmd = resource.ResourceList(self.app, None)
        with open(self.response_path) as f:
            response = json.load(f)
        self.resources = []
        for r in response['resources']:
            self.resources.append(v1_resources.Resource(None, r))

    def test_resource_list(self):
        out = io.StringIO()
        formatter = resource_formatter.ResourceDotFormatter()
        formatter.emit_list(None, self.resources, out, None)
        self.assertEqual(self.data, out.getvalue())