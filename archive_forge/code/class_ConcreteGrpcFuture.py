from concurrent.futures import Future
from typing import Any, Callable, Optional
import pytest
import duet
class ConcreteGrpcFuture(grpc.Future):

    def cancel(self) -> bool:
        return True

    def cancelled(self) -> bool:
        return True

    def running(self) -> bool:
        return True

    def done(self) -> bool:
        return True

    def result(self, timeout: Optional[int]=None) -> Any:
        return 1234

    def exception(self, timeout=None) -> Optional[BaseException]:
        return None

    def add_done_callback(self, fn: Callable[[Any], Any]) -> None:
        pass

    def traceback(self, timeout=None):
        pass