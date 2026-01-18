from saharaclient.api import base
class JobsManagerV2(base.ResourceManager):
    resource_class = Job
    NotUpdated = base.NotUpdated()

    def list(self, search_opts=None, marker=None, limit=None, sort_by=None, reverse=None):
        """Get a list of Jobs."""
        query = base.get_query_string(search_opts, limit=limit, marker=marker, sort_by=sort_by, reverse=reverse)
        url = '/jobs%s' % query
        return self._page(url, 'jobs', limit)

    def get(self, obj_id):
        """Get information about a Job."""
        return self._get('/jobs/%s' % obj_id, 'job')

    def delete(self, obj_id):
        """Delete a Job."""
        self._delete('/jobs/%s' % obj_id)

    def create(self, job_template_id, cluster_id, input_id=None, output_id=None, configs=None, interface=None, is_public=None, is_protected=None):
        """Launch a Job."""
        data = {'cluster_id': cluster_id, 'job_template_id': job_template_id}
        self._copy_if_defined(data, input_id=input_id, output_id=output_id, job_configs=configs, interface=interface, is_public=is_public, is_protected=is_protected)
        return self._create('/jobs', data, 'job')

    def refresh_status(self, obj_id):
        """Refresh Job Status."""
        return self._get('/jobs/%s?refresh_status=True' % obj_id, 'job')

    def update(self, obj_id, is_public=NotUpdated, is_protected=NotUpdated):
        """Update a Job."""
        data = {}
        self._copy_if_updated(data, is_public=is_public, is_protected=is_protected)
        return self._patch('/jobs/%s' % obj_id, data)