from onnx.reference.op_run import OpRun
class Split_2(CommonSplit):

    def _run(self, mat, axis=None, split=None):
        return self.common_run(mat, split, axis=axis, num_outputs=None)