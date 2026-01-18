class PredictionClient(object):
    """A client for Prediction.

  No assumptions are made about whether the prediction happens in process,
  across processes, or even over the network.

  The inputs, unlike Model.predict, have already been "columnarized", i.e.,
  a dict mapping input names to values for a whole batch, much like
  Session.run's feed_dict parameter. The return value is the same format.
  """

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, inputs, **kwargs):
        """Produces predictions for the given inputs.

    Args:
      inputs: A dict mapping input names to values.
      **kwargs: Additional keyword arguments for prediction

    Returns:
      A dict mapping output names to output values, similar to the input
      dict.
    """
        raise NotImplementedError()

    def explain(self, inputs, **kwargs):
        """Produces predictions for the given inputs.

    Args:
      inputs: A dict mapping input names to values.
      **kwargs: Additional keyword arguments for prediction

    Returns:
      A dict mapping output names to output values, similar to the input
      dict.
    """
        raise NotImplementedError()