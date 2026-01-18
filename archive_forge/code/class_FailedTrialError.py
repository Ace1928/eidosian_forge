from keras_tuner.src.api_export import keras_tuner_export
@keras_tuner_export(['keras_tuner.errors.FailedTrialError'])
class FailedTrialError(Exception):
    """Raise this error to mark a `Trial` as failed.

    When this error is raised in a `Trial`, the `Tuner` would not retry the
    `Trial` but directly mark it as `"FAILED"`.

    Example:

    ```py
    class MyHyperModel(keras_tuner.HyperModel):
        def build(self, hp):
            # Build the model
            ...
            if too_slow(model):
                # Mark the Trial as "FAILED" if the model is too slow.
                raise keras_tuner.FailedTrialError("Model is too slow.")
            return model
    ```
    """
    pass