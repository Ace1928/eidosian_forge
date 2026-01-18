from onnx.reference.op_run import OpRun
class OptionalGetElement(OpRun):

    def _run(self, x):
        if x is None:
            raise ValueError('The requested optional input has no value.')
        return (x,)