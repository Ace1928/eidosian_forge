from onnx.reference.op_run import OpRun
class SequenceConstruct(OpRun):

    def _run(self, *data):
        return (list(data),)