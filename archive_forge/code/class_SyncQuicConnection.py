import dns._features
import dns.asyncbackend
class SyncQuicConnection:

    def make_stream(self) -> Any:
        raise NotImplementedError