import enum
class ActionHandle(object):
    id_counter = 0

    def __init__(self, error=False, explanation=''):
        """Constructor"""
        if error:
            self.id = -1
        else:
            self.id = ActionHandle.id_counter
            ActionHandle.id_counter += 1
            self.status = ActionStatus.error
        self.explanation = explanation

    def update(self, ah):
        """Update the contents of the provided ActionHandle"""
        self.id = ah.id
        self.status = ah.status

    def __lt__(self, other):
        return self.id < other.id

    def __hash__(self):
        return self.id.__hash__()

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.id.__hash__() == other.__hash__() and (self.id == other.id)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.id)