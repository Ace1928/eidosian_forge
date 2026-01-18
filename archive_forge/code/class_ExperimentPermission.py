class ExperimentPermission:

    def __init__(self, experiment_id, user_id, permission):
        self._experiment_id = experiment_id
        self._user_id = user_id
        self._permission = permission

    @property
    def experiment_id(self):
        return self._experiment_id

    @property
    def user_id(self):
        return self._user_id

    @property
    def permission(self):
        return self._permission

    @permission.setter
    def permission(self, permission):
        self._permission = permission

    def to_json(self):
        return {'experiment_id': self.experiment_id, 'user_id': self.user_id, 'permission': self.permission}

    @classmethod
    def from_json(cls, dictionary):
        return cls(experiment_id=dictionary['experiment_id'], user_id=dictionary['user_id'], permission=dictionary['permission'])