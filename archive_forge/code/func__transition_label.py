import copy
import abc
import logging
import six
def _transition_label(self, tran):
    edge_label = tran.get('label', tran['trigger'])
    if 'dest' not in tran:
        edge_label += ' [internal]'
    if self.machine.show_conditions and any((prop in tran for prop in ['conditions', 'unless'])):
        edge_label = '{edge_label} [{conditions}]'.format(edge_label=edge_label, conditions=' & '.join(tran.get('conditions', []) + ['!' + u for u in tran.get('unless', [])]))
    return edge_label