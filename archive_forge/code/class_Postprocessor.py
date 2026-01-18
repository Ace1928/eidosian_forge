class Postprocessor(object):
    """Interface for processing a list of instances after prediction."""

    def postprocess(self, instances, **kwargs):
        """The postprocessing function.

    Args:
      instances: A list of instances, as provided to the predict() method.
      **kwargs: Additional keyword arguments for postprocessing.

    Returns:
      The processed instance to return as the final prediction output.
    """
        raise NotImplementedError()