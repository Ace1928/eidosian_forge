import paramiko
import paramiko.client
class _ParamikoWrapper:

    def __init__(self, client, channel) -> None:
        self.client = client
        self.channel = channel
        self.channel.setblocking(True)

    @property
    def stderr(self):
        return self.channel.makefile_stderr('rb')

    def can_read(self):
        return self.channel.recv_ready()

    def write(self, data):
        return self.channel.sendall(data)

    def read(self, n=None):
        data = self.channel.recv(n)
        data_len = len(data)
        if not data:
            return b''
        if n and data_len < n:
            diff_len = n - data_len
            return data + self.read(diff_len)
        return data

    def close(self):
        self.channel.close()