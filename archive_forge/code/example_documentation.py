from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_default
Generates an example based on a shape

        :param section: The section to write the documentation to.

        :param shape: The shape of the operation.

        :param prefix: Anything to be included before the example

        :type include: Dictionary where keys are parameter names and
            values are the shapes of the parameter names.
        :param include: The parameter shapes to include in the documentation.

        :type exclude: List of the names of the parameters to exclude.
        :param exclude: The names of the parameters to exclude from
            documentation.
        