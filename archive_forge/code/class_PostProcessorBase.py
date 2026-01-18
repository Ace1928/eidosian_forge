from nbconvert.utils.base import NbConvertBase
class PostProcessorBase(NbConvertBase):
    """The base class for post processors."""

    def __call__(self, input_):
        """
        See def postprocess() ...
        """
        self.postprocess(input_)

    def postprocess(self, input_):
        """
        Post-process output from a writer.
        """
        msg = 'postprocess'
        raise NotImplementedError(msg)