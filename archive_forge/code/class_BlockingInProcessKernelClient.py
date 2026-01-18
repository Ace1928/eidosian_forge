import sys
from queue import Empty, Queue
from traitlets import Type
from .channels import InProcessChannel
from .client import InProcessKernelClient
class BlockingInProcessKernelClient(InProcessKernelClient):
    """A blocking in-process kernel client."""
    shell_channel_class = Type(BlockingInProcessChannel)
    iopub_channel_class = Type(BlockingInProcessChannel)
    stdin_channel_class = Type(BlockingInProcessStdInChannel)

    def wait_for_ready(self):
        """Wait for kernel info reply on shell channel."""
        while True:
            self.kernel_info()
            try:
                msg = self.shell_channel.get_msg(block=True, timeout=1)
            except Empty:
                pass
            else:
                if msg['msg_type'] == 'kernel_info_reply':
                    try:
                        self.iopub_channel.get_msg(block=True, timeout=0.2)
                    except Empty:
                        pass
                    else:
                        self._handle_kernel_info_reply(msg)
                        break
        while True:
            try:
                msg = self.iopub_channel.get_msg(block=True, timeout=0.2)
                print(msg['msg_type'])
            except Empty:
                break