import boto.vendored.regions.regions as _regions
def _handle_special_cases(self, static_endpoints):
    if 'cloudsearch' in static_endpoints:
        cloudsearch_endpoints = static_endpoints['cloudsearch']
        static_endpoints['cloudsearchdomain'] = cloudsearch_endpoints