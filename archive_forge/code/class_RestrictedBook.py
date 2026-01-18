import statemachine
import librarybookstate
class RestrictedBook(Book):

    def __init__(self):
        super(RestrictedBook, self).__init__()
        self._authorized_users = []

    def authorize(self, name):
        self._authorized_users.append(name)

    def checkout(self, user=None):
        if user in self._authorized_users:
            super().checkout()
        else:
            raise Exception('{0} could not check out restricted book'.format(user if user is not None else 'anonymous'))