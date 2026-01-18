import abc
import enum
import threading  # pylint: disable=unused-import
class Servicer(abc.ABC):
    """Interface for service implementations."""

    @abc.abstractmethod
    def service(self, group, method, context, output_operator):
        """Services an operation.

        Args:
          group: The group identifier of the operation to be serviced.
          method: The method identifier of the operation to be serviced.
          context: An OperationContext object affording contextual information and
            actions.
          output_operator: An Operator that will accept output values of the
            operation.

        Returns:
          A Subscription via which this object may or may not accept more values of
            the operation.

        Raises:
          NoSuchMethodError: If this Servicer does not handle operations with the
            given group and method.
          abandonment.Abandoned: If the operation has been aborted and there no
            longer is any reason to service the operation.
        """
        raise NotImplementedError()