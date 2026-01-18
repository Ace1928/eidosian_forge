import os.path as op
import networkx as nx
from nipype import Workflow, MapNode, Node, IdentityInterface
from nipype.interfaces.base import (  # BaseInterfaceInputSpec,
class OutputWrapper(object):
    """Implements connections between workflow outputs and DataSink."""

    def __init__(self, workflow, subject_node, sink_node, output_node):
        self.wf = workflow
        self.subj_node = subject_node
        self.sink_node = sink_node
        self.out_node = output_node

    def set_mapnode_substitutions(self, n_runs, template_pattern='run_%d', template_args='r + 1'):
        """Find mapnode names and add datasink substitutions to sort by run."""
        mapnode_names = find_mapnodes(self.wf)
        nested_workflows = find_nested_workflows(self.wf)
        for wf in nested_workflows:
            mapnode_names.extend(find_mapnodes(wf))
        substitutions = []
        for r in reversed(range(n_runs)):
            templ_args = eval('(%s)' % template_args)
            for name in mapnode_names:
                substitutions.append(('_%s%d' % (name, r), template_pattern % templ_args))
        if isdefined(self.sink_node.inputs.substitutions):
            self.sink_node.inputs.substitutions.extend(substitutions)
        else:
            self.sink_node.inputs.substitutions = substitutions

    def add_regexp_substitutions(self, sub_list):
        """Safely set subsitutions implemented with regular expressions."""
        if isdefined(self.sink_node.inputs.regexp_substitutions):
            self.sink_node.inputs.regexp_substitutions.extend(sub_list)
        else:
            self.sink_node.inputs.regexp_substitutions = sub_list

    def set_subject_container(self):
        """Store results by subject at highest level."""
        self.wf.connect(self.subj_node, 'subject_id', self.sink_node, 'container')
        subj_subs = []
        for s in self.subj_node.iterables[1]:
            subj_subs.append(('/_subject_id_%s/' % s, '/'))
        if isdefined(self.sink_node.inputs.substitutions):
            self.sink_node.inputs.substitutions.extend(subj_subs)
        else:
            self.sink_node.inputs.substitutions = subj_subs

    def sink_outputs(self, dir_name=None):
        """Connect the outputs of a workflow to a datasink."""
        outputs = self.out_node.outputs.get()
        prefix = '@' if dir_name is None else dir_name + '.@'
        for field in outputs:
            self.wf.connect(self.out_node, field, self.sink_node, prefix + field)