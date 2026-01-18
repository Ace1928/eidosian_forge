from tensorflow.python.util.tf_export import tf_export
A mixin that delegates all Trackable methods to another trackable object.

  DO NOT USE THIS UNLESS YOU ARE THE KERAS LOSS SCALE OPTIMIZER.

  This class must be used with multiple inheritance. A class that subclasses
  Trackable can also subclass this class, which causes all Trackable methods to
  be delegated to the trackable object passed in the constructor.

  A subclass can use this mixin to appear as if it were the trackable passed to
  the constructor, from a Checkpoint's perspective. LossScaleOptimizer uses this
  mixin, so that the checkpoint format for a LossScaleOptimizer is identical to
  the checkpoint format for a normal optimizer. This allows a model to be saved
  with a normal Optimizer and restored with a LossScaleOptimizer, or vice versa.
  The only difference in checkpoint format is that the loss scale is also saved
  with a LossScaleOptimizer.
  