import ast
import io
import sys
import tokenize
def __For_helper(self, fill, t):
    self.fill(fill)
    self.dispatch(t.target)
    self.write(' in ')
    self.dispatch(t.iter)
    self.enter()
    self.dispatch(t.body)
    self.leave()
    if t.orelse:
        self.fill('else')
        self.enter()
        self.dispatch(t.orelse)
        self.leave()