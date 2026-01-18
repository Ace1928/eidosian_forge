import torch.distributed.rpc as rpc
import torch.distributed.rpc._testing  # noqa: F401
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
class FaultyRpcAgentTestFixture(RpcAgentTestFixture):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages_to_fail = retryable_message_types
        self.messages_to_delay = default_messages_to_delay

    @property
    def rpc_backend(self):
        return rpc.backend_registry.BackendType['FAULTY_TENSORPIPE']

    @property
    def rpc_backend_options(self):
        return rpc.backend_registry.construct_rpc_backend_options(self.rpc_backend, init_method=self.init_method, num_worker_threads=8, num_fail_sends=3, messages_to_fail=self.messages_to_fail, messages_to_delay=self.messages_to_delay)

    def setup_fault_injection(self, faulty_messages, messages_to_delay):
        if faulty_messages is not None:
            self.messages_to_fail = faulty_messages
        if messages_to_delay is not None:
            self.messages_to_delay = messages_to_delay

    def get_shutdown_error_regex(self):
        error_regexes = ['Exception in thread pool task', 'Connection reset by peer', 'Connection closed by peer']
        return '|'.join([f'({error_str})' for error_str in error_regexes])

    def get_timeout_error_regex(self):
        return 'RPC ran for more than'