from oslo_messaging.notify import notifier
class NoOpDriver(notifier.Driver):

    def notify(self, ctxt, message, priority, retry):
        pass