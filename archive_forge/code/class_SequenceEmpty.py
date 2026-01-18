from onnx.reference.op_run import OpRun
class SequenceEmpty(OpRun):

    def _run(self, dtype=None):
        return ([],)