from designateclient.v2.base import V2Controller
class QuotasController(V2Controller):

    def list(self, project_id):
        return self._get(f'/quotas/{project_id}')

    def update(self, project_id, values):
        return self._patch(f'/quotas/{project_id}', data=values)

    def reset(self, project_id):
        return self._delete(f'/quotas/{project_id}')