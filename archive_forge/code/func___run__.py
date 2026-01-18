from __future__ import absolute_import, division, print_function
def __run__(self):
    state = self._state()
    self.vars.state = state
    if state not in self.module.params:
        aliased = [name for name, param in self.module.argument_spec.items() if state in param.get('aliases', [])]
        if aliased:
            state = aliased[0]
            self.vars.effective_state = state
    method = self._method(state)
    if not hasattr(self, method):
        return self.__state_fallback__()
    func = getattr(self, method)
    return func()