from __future__ import absolute_import, division, print_function
import argparse
import ast
import os
import re
import requests
import sys
from time import time
import json
class DoManager:

    def __init__(self, api_token):
        self.api_token = api_token
        self.api_endpoint = 'https://api.digitalocean.com/v2'
        self.headers = {'Authorization': 'Bearer {0}'.format(self.api_token), 'Content-type': 'application/json'}
        self.timeout = 60

    def _url_builder(self, path):
        if path[0] == '/':
            path = path[1:]
        return '%s/%s' % (self.api_endpoint, path)

    def send(self, url, method='GET', data=None, params=None):
        url = self._url_builder(url)
        data = json.dumps(data)
        try:
            if method == 'GET':
                resp_data = {}
                incomplete = True
                while incomplete:
                    resp = requests.get(url, data=data, params=params, headers=self.headers, timeout=self.timeout)
                    json_resp = resp.json()
                    for key, value in json_resp.items():
                        if isinstance(value, list) and key in resp_data:
                            resp_data[key] += value
                        else:
                            resp_data[key] = value
                    try:
                        url = json_resp['links']['pages']['next']
                    except KeyError:
                        incomplete = False
        except ValueError as e:
            sys.exit('Unable to parse result from %s: %s' % (url, e))
        return resp_data

    def all_active_droplets(self, tag_name=None):
        if tag_name is not None:
            params = {'tag_name': tag_name}
            resp = self.send('droplets/', params=params)
        else:
            resp = self.send('droplets/')
        return resp['droplets']

    def all_regions(self):
        resp = self.send('regions/')
        return resp['regions']

    def all_images(self, filter_name='global'):
        params = {'filter': filter_name}
        resp = self.send('images/', data=params)
        return resp['images']

    def sizes(self):
        resp = self.send('sizes/')
        return resp['sizes']

    def all_ssh_keys(self):
        resp = self.send('account/keys')
        return resp['ssh_keys']

    def all_domains(self):
        resp = self.send('domains/')
        return resp['domains']

    def show_droplet(self, droplet_id):
        resp = self.send('droplets/%s' % droplet_id)
        return resp.get('droplet', {})

    def all_tags(self):
        resp = self.send('tags')
        return resp['tags']