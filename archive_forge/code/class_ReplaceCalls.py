from llvmlite.ir import CallInstr
class ReplaceCalls(CallVisitor):

    def __init__(self, orig, repl):
        super(ReplaceCalls, self).__init__()
        self.orig = orig
        self.repl = repl
        self.calls = []

    def visit_Call(self, instr):
        if instr.callee == self.orig:
            instr.replace_callee(self.repl)
            self.calls.append(instr)