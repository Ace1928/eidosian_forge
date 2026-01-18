import copy
import abc
import logging
import six
def _get_elements(self):
    states = []
    transitions = []
    try:
        markup = self.machine.get_markup_config()
        queue = [([], markup)]
        while queue:
            prefix, scope = queue.pop(0)
            for transition in scope.get('transitions', []):
                if prefix:
                    tran = copy.copy(transition)
                    tran['source'] = self.machine.state_cls.separator.join(prefix + [tran['source']])
                    if 'dest' in tran:
                        tran['dest'] = self.machine.state_cls.separator.join(prefix + [tran['dest']])
                else:
                    tran = transition
                transitions.append(tran)
            for state in scope.get('children', []) + scope.get('states', []):
                if not prefix:
                    sta = state
                    states.append(sta)
                ini = state.get('initial', [])
                if not isinstance(ini, list):
                    ini = ini.name if hasattr(ini, 'name') else ini
                    tran = dict(trigger='', source=self.machine.state_cls.separator.join(prefix + [state['name']]) + '_anchor', dest=self.machine.state_cls.separator.join(prefix + [state['name'], ini]))
                    transitions.append(tran)
                if state.get('children', []):
                    queue.append((prefix + [state['name']], state))
    except KeyError:
        _LOGGER.error('Graph creation incomplete!')
    return (states, transitions)