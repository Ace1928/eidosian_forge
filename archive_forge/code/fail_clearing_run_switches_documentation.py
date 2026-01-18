import greenlet

If we have a run callable passed to the constructor or set as an
attribute, but we don't actually use that (because ``__getattribute__``
or the like interferes), then when we clear callable before beginning
to run, there's an opportunity for Python code to run.

