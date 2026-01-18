from saharaclient.api import base
class JobsManagerV1(base.ResourceManager):
    resource_class = Job
    NotUpdated = base.NotUpdated()

    def create(self, name, type, mains=None, libs=None, description=None, interface=None, is_public=None, is_protected=None):
        """Create a Job."""
        data = {'name': name, 'type': type}
        self._copy_if_defined(data, description=description, mains=mains, libs=libs, interface=interface, is_public=is_public, is_protected=is_protected)
        return self._create('/jobs', data, 'job')

    def list(self, search_opts=None, limit=None, marker=None, sort_by=None, reverse=None):
        """Get a list of Jobs."""
        query = base.get_query_string(search_opts, limit=limit, marker=marker, sort_by=sort_by, reverse=reverse)
        url = '/jobs%s' % query
        return self._page(url, 'jobs', limit)

    def get(self, job_id):
        """Get information about a Job"""
        return self._get('/jobs/%s' % job_id, 'job')

    def get_configs(self, job_type):
        """Get config hints for a specified Job type."""
        return self._get('/jobs/config-hints/%s' % job_type)

    def delete(self, job_id):
        """Delete a Job"""
        self._delete('/jobs/%s' % job_id)

    def update(self, job_id, name=NotUpdated, description=NotUpdated, is_public=NotUpdated, is_protected=NotUpdated):
        """Update a Job."""
        data = {}
        self._copy_if_updated(data, name=name, description=description, is_public=is_public, is_protected=is_protected)
        return self._patch('/jobs/%s' % job_id, data)