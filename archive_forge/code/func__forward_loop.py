import logging
import threading
def _forward_loop(self):
    """read forwarding requests from the queue"""
    while True:
        try:
            digest, whitelist = self.forward_queue.get(block=True, timeout=2)
        except Queue.Empty:
            if self.forwarding_client is None:
                return
            else:
                continue
        for server in self.remote_servers:
            try:
                if whitelist:
                    self.forwarding_client.whitelist(digest, server)
                else:
                    self.forwarding_client.report(digest, server)
            except Exception as ex:
                self.log.warn('Forwarding digest %s to %s failed: %s', digest, server, ex)