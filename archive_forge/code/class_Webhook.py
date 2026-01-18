class Webhook(object):
    url = 'v1/webhook/'

    def __init__(self, api):
        self.api = api

    def list(self, all_tenants=False):
        """Get webhook list"""
        params = dict(all_tenants=all_tenants)
        return self.api.get(self.url, params=params).json()

    def show(self, id):
        """Show specific webhook

        :param id: id of registration to show
        """
        url = self.url + id
        return self.api.get(url).json()

    def add(self, url, regex_filter=None, headers=None):
        """Add a webhook to the DB


        :param url: url to register in the DB
        :param regex_filter: a optional regular expression dict to filter
        alarms
        :param headers: optional headers to attach to requests
        """
        params = dict(url=url, regex_filter=regex_filter, headers=headers)
        return self.api.post(self.url, json=params).json()

    def delete(self, id):
        """delete a webhook from the DB


        :param id: id of webhook to delete
        """
        url = self.url + id
        return self.api.delete(url).json()