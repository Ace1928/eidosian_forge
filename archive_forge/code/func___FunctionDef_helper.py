import ast
import io
import sys
import tokenize
def __FunctionDef_helper(self, t, fill_suffix):
    self.write('\n')
    for deco in t.decorator_list:
        self.fill('@')
        self.dispatch(deco)
    def_str = fill_suffix + ' ' + t.name + '('
    self.fill(def_str)
    self.dispatch(t.args)
    self.write(')')
    if t.returns:
        self.write(' -> ')
        self.dispatch(t.returns)
    self.enter()
    self.dispatch(t.body)
    self.leave()