from ncclient.transport import SessionListener
class PrintListener(SessionListener):

    def callback(self, root, raw):
        print('\n# RECEIVED MESSAGE with root=[tag=%r, attrs=%r] #\n%r\n' % (root[0], root[1], raw))

    def errback(self, err):
        print('\n# RECEIVED ERROR #\n%r\n' % err)