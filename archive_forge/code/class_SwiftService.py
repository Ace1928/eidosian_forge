import logging
import os
from collections import defaultdict
from concurrent.futures import as_completed, CancelledError, TimeoutError
from copy import deepcopy
from errno import EEXIST, ENOENT
from hashlib import md5
from io import StringIO
from os import environ, makedirs, stat, utime
from os.path import (
from posixpath import join as urljoin
from random import shuffle
from time import time
from threading import Thread
from queue import Queue
from queue import Empty as QueueEmpty
from urllib.parse import quote
import json
from swiftclient import Connection
from swiftclient.command_helpers import (
from swiftclient.utils import (
from swiftclient.exceptions import ClientException
from swiftclient.multithreading import MultiThreadingManager
class SwiftService:
    """
    Service for performing swift operations
    """

    def __init__(self, options=None):
        if options is not None:
            self._options = dict(_default_global_options, **dict(_default_local_options, **options))
        else:
            self._options = dict(_default_global_options, **_default_local_options)
        process_options(self._options)

        def create_connection():
            return get_conn(self._options)
        self.thread_manager = MultiThreadingManager(create_connection, segment_threads=self._options['segment_threads'], object_dd_threads=self._options['object_dd_threads'], object_uu_threads=self._options['object_uu_threads'], container_threads=self._options['container_threads'])
        self.capabilities_cache = {}

    def __enter__(self):
        self.thread_manager.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.thread_manager.__exit__(exc_type, exc_val, exc_tb)

    def stat(self, container=None, objects=None, options=None):
        """
        Get account stats, container stats or information about a list of
        objects in a container.

        :param container: The container to query.
        :param objects: A list of object paths about which to return
                        information (a list of strings).
        :param options: A dictionary containing options to override the global
                        options specified during the service object creation.
                        These options are applied to all stat operations
                        performed by this call::

                            {
                                'human': False,
                                'version_id': None,
                                'header': []
                            }

        :returns: Either a single dictionary containing stats about an account
                  or container, or an iterator for returning the results of the
                  stat operations on a list of objects.

        :raises SwiftError:
        """
        if options is not None:
            options = dict(self._options, **options)
        else:
            options = self._options
        if not container:
            if objects:
                raise SwiftError('Objects specified without container')
            else:
                res = {'action': 'stat_account', 'success': True, 'container': container, 'object': None}
                try:
                    stats_future = self.thread_manager.container_pool.submit(stat_account, options)
                    items, headers = get_future_result(stats_future)
                    res.update({'items': items, 'headers': headers})
                    return res
                except ClientException as err:
                    if err.http_status != 404:
                        traceback, err_time = report_traceback()
                        logger.exception(err)
                        res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time})
                        return res
                    raise SwiftError('Account not found', exc=err)
                except Exception as err:
                    traceback, err_time = report_traceback()
                    logger.exception(err)
                    res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time})
                    return res
        elif not objects:
            res = {'action': 'stat_container', 'container': container, 'object': None, 'success': True}
            try:
                stats_future = self.thread_manager.container_pool.submit(stat_container, options, container)
                items, headers = get_future_result(stats_future)
                res.update({'items': items, 'headers': headers})
                return res
            except ClientException as err:
                if err.http_status != 404:
                    traceback, err_time = report_traceback()
                    logger.exception(err)
                    res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time})
                    return res
                raise SwiftError('Container %r not found' % container, container=container, exc=err)
            except Exception as err:
                traceback, err_time = report_traceback()
                logger.exception(err)
                res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time})
                return res
        else:
            stat_futures = []
            for stat_o in objects:
                stat_future = self.thread_manager.object_dd_pool.submit(self._stat_object, container, stat_o, options)
                stat_futures.append(stat_future)
            return ResultsIterator(stat_futures)

    @staticmethod
    def _stat_object(conn, container, obj, options):
        res = {'action': 'stat_object', 'object': obj, 'container': container, 'success': True}
        try:
            items, headers = stat_object(conn, options, container, obj)
            res.update({'items': items, 'headers': headers})
            return res
        except Exception as err:
            traceback, err_time = report_traceback()
            logger.exception(err)
            res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time})
            return res

    def post(self, container=None, objects=None, options=None):
        """
        Post operations on an account, container or list of objects

        :param container: The container to make the post operation against.
        :param objects: A list of object names (strings) or SwiftPostObject
                        instances containing an object name, and an
                        options dict (can be None) to override the options for
                        that individual post operation::

                            [
                                'object_name',
                                SwiftPostObject('object_name', options={...}),
                                ...
                            ]

                        The options dict is described below.
        :param options: A dictionary containing options to override the global
                        options specified during the service object creation.
                        These options are applied to all post operations
                        performed by this call, unless overridden on a per
                        object basis. Possible options are given below::

                            {
                                'meta': [],
                                'header': [],
                                'read_acl': None,   # For containers only
                                'write_acl': None,  # For containers only
                                'sync_to': None,    # For containers only
                                'sync_key': None    # For containers only
                            }

        :returns: Either a single result dictionary in the case of a post to a
                  container/account, or an iterator for returning the results
                  of posts to a list of objects.

        :raises SwiftError:
        """
        if options is not None:
            options = dict(self._options, **options)
        else:
            options = self._options
        res = {'success': True, 'container': container, 'object': None, 'headers': {}}
        if not container:
            res['action'] = 'post_account'
            if objects:
                raise SwiftError('Objects specified without container')
            else:
                response_dict = {}
                headers = split_headers(options['meta'], 'X-Account-Meta-')
                headers.update(split_headers(options['header'], ''))
                res['headers'] = headers
                try:
                    post = self.thread_manager.container_pool.submit(self._post_account_job, headers, response_dict)
                    get_future_result(post)
                except ClientException as err:
                    if err.http_status != 404:
                        traceback, err_time = report_traceback()
                        logger.exception(err)
                        res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time, 'response_dict': response_dict})
                        return res
                    raise SwiftError('Account not found', exc=err)
                except Exception as err:
                    traceback, err_time = report_traceback()
                    logger.exception(err)
                    res.update({'success': False, 'error': err, 'response_dict': response_dict, 'traceback': traceback, 'error_timestamp': err_time})
            return res
        if not objects:
            res['action'] = 'post_container'
            response_dict = {}
            headers = split_headers(options['meta'], 'X-Container-Meta-')
            headers.update(split_headers(options['header'], ''))
            if options['read_acl'] is not None:
                headers['X-Container-Read'] = options['read_acl']
            if options['write_acl'] is not None:
                headers['X-Container-Write'] = options['write_acl']
            if options['sync_to'] is not None:
                headers['X-Container-Sync-To'] = options['sync_to']
            if options['sync_key'] is not None:
                headers['X-Container-Sync-Key'] = options['sync_key']
            res['headers'] = headers
            try:
                post = self.thread_manager.container_pool.submit(self._post_container_job, container, headers, response_dict)
                get_future_result(post)
            except ClientException as err:
                if err.http_status != 404:
                    traceback, err_time = report_traceback()
                    logger.exception(err)
                    res.update({'action': 'post_container', 'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time, 'response_dict': response_dict})
                    return res
                raise SwiftError("Container '%s' not found" % container, container=container, exc=err)
            except Exception as err:
                traceback, err_time = report_traceback()
                logger.exception(err)
                res.update({'action': 'post_container', 'success': False, 'error': err, 'response_dict': response_dict, 'traceback': traceback, 'error_timestamp': err_time})
            return res
        else:
            post_futures = []
            post_objects = self._make_post_objects(objects)
            for post_object in post_objects:
                obj = post_object.object_name
                obj_options = post_object.options
                response_dict = {}
                headers = split_headers(options['meta'], 'X-Object-Meta-')
                headers.update(split_headers(options['header'], ''))
                if obj_options is not None:
                    if 'meta' in obj_options:
                        headers.update(split_headers(obj_options['meta'], 'X-Object-Meta-'))
                    if 'header' in obj_options:
                        headers.update(split_headers(obj_options['header'], ''))
                post = self.thread_manager.object_uu_pool.submit(self._post_object_job, container, obj, headers, response_dict)
                post_futures.append(post)
            return ResultsIterator(post_futures)

    @staticmethod
    def _make_post_objects(objects):
        post_objects = []
        for o in objects:
            if isinstance(o, str):
                obj = SwiftPostObject(o)
                post_objects.append(obj)
            elif isinstance(o, SwiftPostObject):
                post_objects.append(o)
            else:
                raise SwiftError('The post operation takes only strings or SwiftPostObjects as input', obj=o)
        return post_objects

    @staticmethod
    def _post_account_job(conn, headers, result):
        return conn.post_account(headers=headers, response_dict=result)

    @staticmethod
    def _post_container_job(conn, container, headers, result):
        try:
            res = conn.post_container(container, headers=headers, response_dict=result)
        except ClientException as err:
            if err.http_status != 404:
                raise
            _response_dict = {}
            res = conn.put_container(container, headers=headers, response_dict=_response_dict)
            result['post_put'] = _response_dict
        return res

    @staticmethod
    def _post_object_job(conn, container, obj, headers, result):
        res = {'success': True, 'action': 'post_object', 'container': container, 'object': obj, 'headers': headers, 'response_dict': result}
        try:
            conn.post_object(container, obj, headers=headers, response_dict=result)
        except Exception as err:
            traceback, err_time = report_traceback()
            logger.exception(err)
            res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time})
        return res

    def list(self, container=None, options=None):
        """
        List operations on an account, container.

        :param container: The container to make the list operation against.
        :param options: A dictionary containing options to override the global
                        options specified during the service object creation::

                            {
                                'long': False,
                                'prefix': None,
                                'delimiter': None,
                                'versions': False,
                                'header': []
                            }

        :returns: A generator for returning the results of the list operation
                  on an account or container. Each result yielded from the
                  generator is either a 'list_account_part' or
                  'list_container_part', containing part of the listing.
        """
        if options is not None:
            options = dict(self._options, **options)
        else:
            options = self._options
        rq = Queue(maxsize=10)
        if container is None:
            listing_future = self.thread_manager.container_pool.submit(self._list_account_job, options, rq)
        else:
            listing_future = self.thread_manager.container_pool.submit(self._list_container_job, container, options, rq)
        res = get_from_queue(rq)
        while res is not None:
            yield res
            res = get_from_queue(rq)
        get_future_result(listing_future)

    @staticmethod
    def _list_account_job(conn, options, result_queue):
        marker = ''
        error = None
        req_headers = split_headers(options.get('header', []))
        try:
            while True:
                _, items = conn.get_account(marker=marker, prefix=options['prefix'], headers=req_headers)
                if not items:
                    result_queue.put(None)
                    return
                if options['long']:
                    for i in items:
                        name = i['name']
                        i['meta'] = conn.head_container(name)
                res = {'action': 'list_account_part', 'container': None, 'prefix': options['prefix'], 'success': True, 'listing': items, 'marker': marker}
                result_queue.put(res)
                marker = items[-1].get('name', items[-1].get('subdir'))
        except ClientException as err:
            traceback, err_time = report_traceback()
            logger.exception(err)
            if err.http_status != 404:
                error = (err, traceback, err_time)
            else:
                error = (SwiftError('Account not found', exc=err), traceback, err_time)
        except Exception as err:
            traceback, err_time = report_traceback()
            logger.exception(err)
            error = (err, traceback, err_time)
        res = {'action': 'list_account_part', 'container': None, 'prefix': options['prefix'], 'success': False, 'marker': marker, 'error': error[0], 'traceback': error[1], 'error_timestamp': error[2]}
        result_queue.put(res)
        result_queue.put(None)

    @staticmethod
    def _list_container_job(conn, container, options, result_queue):
        marker = options.get('marker', '')
        version_marker = options.get('version_marker', '')
        error = None
        req_headers = split_headers(options.get('header', []))
        if options.get('versions', False):
            query_string = 'versions=true'
        else:
            query_string = None
        try:
            while True:
                _, items = conn.get_container(container, marker=marker, version_marker=version_marker, prefix=options['prefix'], delimiter=options['delimiter'], headers=req_headers, query_string=query_string)
                if not items:
                    result_queue.put(None)
                    return
                res = {'action': 'list_container_part', 'container': container, 'prefix': options['prefix'], 'success': True, 'marker': marker, 'listing': items}
                result_queue.put(res)
                marker = items[-1].get('name', items[-1].get('subdir'))
                version_marker = items[-1].get('version_id', '')
        except ClientException as err:
            traceback, err_time = report_traceback()
            if err.http_status != 404:
                logger.exception(err)
                error = (err, traceback, err_time)
            else:
                error = (SwiftError('Container %r not found' % container, container=container, exc=err), traceback, err_time)
        except Exception as err:
            traceback, err_time = report_traceback()
            logger.exception(err)
            error = (err, traceback, err_time)
        res = {'action': 'list_container_part', 'container': container, 'prefix': options['prefix'], 'success': False, 'marker': marker, 'version_marker': version_marker, 'error': error[0], 'traceback': error[1], 'error_timestamp': error[2]}
        result_queue.put(res)
        result_queue.put(None)

    def download(self, container=None, objects=None, options=None):
        """
        Download operations on an account, optional container and optional list
        of objects.

        :param container: The container to download from.
        :param objects: A list of object names to download (a list of strings).
        :param options: A dictionary containing options to override the global
                        options specified during the service object creation::

                            {
                                'yes_all': False,
                                'marker': '',
                                'prefix': None,
                                'no_download': False,
                                'header': [],
                                'skip_identical': False,
                                'version_id': None,
                                'out_directory': None,
                                'checksum': True,
                                'out_file': None,
                                'remove_prefix': False,
                                'shuffle' : False
                            }

        :returns: A generator for returning the results of the download
                  operations. Each result yielded from the generator is a
                  'download_object' dictionary containing the results of an
                  individual file download.

        :raises ClientException:
        :raises SwiftError:
        """
        if options is not None:
            options = dict(self._options, **options)
        else:
            options = self._options
        if not container:
            if options['yes_all']:
                try:
                    options_copy = deepcopy(options)
                    options_copy['long'] = False
                    for part in self.list(options=options_copy):
                        if part['success']:
                            containers = [i['name'] for i in part['listing']]
                            if options['shuffle']:
                                shuffle(containers)
                            for con in containers:
                                for res in self._download_container(con, options_copy):
                                    yield res
                        else:
                            raise part['error']
                except ClientException as err:
                    if err.http_status != 404:
                        raise
                    raise SwiftError('Account not found', exc=err)
        elif objects is None:
            if '/' in container:
                raise SwiftError("'/' in container name", container=container)
            for res in self._download_container(container, options):
                yield res
        else:
            if '/' in container:
                raise SwiftError("'/' in container name", container=container)
            if options['out_file'] and len(objects) > 1:
                options['out_file'] = None
            o_downs = [self.thread_manager.object_dd_pool.submit(self._download_object_job, container, obj, options) for obj in objects]
            for o_down in interruptable_as_completed(o_downs):
                yield o_down.result()

    def _download_object_job(self, conn, container, obj, options):
        out_file = options['out_file']
        results_dict = {}
        req_headers = split_headers(options['header'], '')
        pseudodir = False
        path = join(container, obj) if options['yes_all'] else obj
        path = path.lstrip(os_path_sep)
        options['skip_identical'] = options['skip_identical'] and out_file != '-'
        if options['prefix'] and options['remove_prefix']:
            path = path[len(options['prefix']):].lstrip('/')
        if options['out_directory']:
            path = os.path.join(options['out_directory'], path)
        if options['skip_identical']:
            filename = out_file if out_file else path
            try:
                fp = open(filename, 'rb', DISK_BUFFER)
            except IOError:
                pass
            else:
                with fp:
                    md5sum = md5()
                    while True:
                        data = fp.read(DISK_BUFFER)
                        if not data:
                            break
                        md5sum.update(data)
                    req_headers['If-None-Match'] = md5sum.hexdigest()
        try:
            start_time = time()
            get_args = {'resp_chunk_size': DISK_BUFFER, 'headers': req_headers, 'response_dict': results_dict}
            if options.get('version_id') is not None:
                get_args['query_string'] = 'version-id=%s' % options['version_id']
            if options['skip_identical']:
                get_args['query_string'] = 'multipart-manifest=get'
            try:
                headers, body = conn.get_object(container, obj, **get_args)
            except ClientException as e:
                if not options['skip_identical']:
                    raise
                if e.http_status != 304:
                    raise
                headers = results_dict['headers']
                if 'x-object-manifest' in headers:
                    body = []
                elif config_true_value(headers.get('x-static-large-object')):
                    body = [b'[]']
                else:
                    raise
            if options['skip_identical']:
                if config_true_value(headers.get('x-static-large-object')) or 'x-object-manifest' in headers:
                    chunk_data = self._get_chunk_data(conn, container, obj, headers, b''.join(body))
                else:
                    chunk_data = None
                if chunk_data is not None:
                    if self._is_identical(chunk_data, filename):
                        raise ClientException('Large object is identical', http_status=304)
                    del get_args['query_string']
                    get_args['response_dict'].clear()
                    headers, body = conn.get_object(container, obj, **get_args)
            headers_receipt = time()
            obj_body = _SwiftReader(path, body, headers, options.get('checksum', True))
            no_file = options['no_download']
            if out_file == '-' and (not no_file):
                res = {'action': 'download_object', 'container': container, 'object': obj, 'path': path, 'pseudodir': pseudodir, 'contents': obj_body}
                return res
            fp = None
            try:
                content_type = headers.get('content-type', '').split(';', 1)[0]
                if content_type in KNOWN_DIR_MARKERS:
                    make_dir = not no_file and out_file != '-'
                    if make_dir and (not isdir(path)):
                        mkdirs(path)
                else:
                    make_dir = not (no_file or out_file)
                    if make_dir:
                        dirpath = dirname(path)
                        if dirpath and (not isdir(dirpath)):
                            mkdirs(dirpath)
                    if not no_file:
                        if out_file:
                            fp = open(out_file, 'wb', DISK_BUFFER)
                        elif basename(path):
                            fp = open(path, 'wb', DISK_BUFFER)
                        else:
                            pseudodir = True
                for chunk in obj_body:
                    if fp is not None:
                        fp.write(chunk)
                finish_time = time()
            finally:
                bytes_read = obj_body.bytes_read()
                if fp is not None:
                    fp.close()
                    if 'x-object-meta-mtime' in headers and (not no_file) and (not options['ignore_mtime']):
                        try:
                            mtime = float(headers['x-object-meta-mtime'])
                        except ValueError:
                            pass
                        else:
                            if options['out_file']:
                                utime(options['out_file'], (mtime, mtime))
                            else:
                                utime(path, (mtime, mtime))
            res = {'action': 'download_object', 'success': True, 'container': container, 'object': obj, 'path': path, 'pseudodir': pseudodir, 'start_time': start_time, 'finish_time': finish_time, 'headers_receipt': headers_receipt, 'auth_end_time': conn.auth_end_time, 'read_length': bytes_read, 'attempts': conn.attempts, 'response_dict': results_dict}
            return res
        except Exception as err:
            traceback, err_time = report_traceback()
            logger.exception(err)
            res = {'action': 'download_object', 'container': container, 'object': obj, 'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time, 'response_dict': results_dict, 'path': path, 'pseudodir': pseudodir, 'attempts': conn.attempts}
            return res

    def _submit_page_downloads(self, container, page_generator, options):
        try:
            list_page = next(page_generator)
        except StopIteration:
            return None
        if list_page['success']:
            objects = [o['name'] for o in list_page['listing']]
            if options['shuffle']:
                shuffle(objects)
            o_downs = [self.thread_manager.object_dd_pool.submit(self._download_object_job, container, obj, options) for obj in objects]
            return o_downs
        else:
            raise list_page['error']

    def _download_container(self, container, options):
        _page_generator = self.list(container=container, options=options)
        try:
            next_page_downs = self._submit_page_downloads(container, _page_generator, options)
        except ClientException as err:
            if err.http_status != 404:
                raise
            raise SwiftError('Container %r not found' % container, container=container, exc=err)
        error = None
        while next_page_downs:
            page_downs = next_page_downs
            next_page_downs = None
            next_page_triggered = False
            next_page_trigger_point = 0.8 * len(page_downs)
            page_results_yielded = 0
            for o_down in interruptable_as_completed(page_downs):
                yield o_down.result()
                if not next_page_triggered:
                    page_results_yielded += 1
                    if page_results_yielded >= next_page_trigger_point:
                        try:
                            next_page_downs = self._submit_page_downloads(container, _page_generator, options)
                        except ClientException as err:
                            logger.exception(err)
                            error = err
                        except Exception:
                            for _d in page_downs:
                                _d.cancel()
                            raise
                        finally:
                            next_page_triggered = True
        if error:
            raise error

    def upload(self, container, objects, options=None):
        """
        Upload a list of objects to a given container.

        :param container: The container (or pseudo-folder path) to put the
                          uploads into.
        :param objects: A list of file/directory names (strings) or
                        SwiftUploadObject instances containing a source for the
                        created object, an object name, and an options dict
                        (can be None) to override the options for that
                        individual upload operation::

                            [
                                '/path/to/file',
                                SwiftUploadObject('/path', object_name='obj1'),
                                ...
                            ]

                        The options dict is as described below.

                        The SwiftUploadObject source may be one of:

                            * A file-like object (with a read method)
                            * A string containing the path to a local
                              file or directory
                            * None, to indicate that we want an empty object

        :param options: A dictionary containing options to override the global
                        options specified during the service object creation.
                        These options are applied to all upload operations
                        performed by this call, unless overridden on a per
                        object basis. Possible options are given below::

                            {
                                'meta': [],
                                'header': [],
                                'segment_size': None,
                                'use_slo': True,
                                'segment_container': None,
                                'leave_segments': False,
                                'changed': None,
                                'skip_identical': False,
                                'skip_container_put': False,
                                'fail_fast': False,
                                'dir_marker': False  # Only for None sources
                            }

        :returns: A generator for returning the results of the uploads.

        :raises SwiftError:
        :raises ClientException:
        """
        if options is not None:
            options = dict(self._options, **options)
        else:
            options = self._options
        try:
            segment_size = int(0 if options['segment_size'] is None else options['segment_size'])
        except ValueError:
            raise SwiftError('Segment size should be an integer value')
        if segment_size and options['use_slo'] is None:
            try:
                cap_result = self.capabilities()
            except ClientException:
                options['use_slo'] = False
            else:
                if not cap_result['success']:
                    options['use_slo'] = False
                else:
                    options['use_slo'] = 'slo' in cap_result['capabilities']
        container, _sep, pseudo_folder = container.partition('/')
        if not options['skip_container_put']:
            policy_header = {}
            _header = split_headers(options['header'])
            if POLICY in _header:
                policy_header[POLICY] = _header[POLICY]
            create_containers = [self.thread_manager.container_pool.submit(self._create_container_job, container, headers=policy_header)]
            for r in interruptable_as_completed(create_containers):
                res = r.result()
                yield res
            if segment_size:
                seg_container = container + '_segments'
                if options['segment_container']:
                    seg_container = options['segment_container']
                if seg_container != container:
                    if not policy_header:
                        create_containers = [self.thread_manager.container_pool.submit(self._create_container_job, seg_container, policy_source=container)]
                    else:
                        create_containers = [self.thread_manager.container_pool.submit(self._create_container_job, seg_container, headers=policy_header)]
                    for r in interruptable_as_completed(create_containers):
                        res = r.result()
                        yield res
        rq = Queue()
        file_jobs = {}
        upload_objects = self._make_upload_objects(objects, pseudo_folder)
        for upload_object in upload_objects:
            s = upload_object.source
            o = upload_object.object_name
            o_opts = upload_object.options
            details = {'action': 'upload', 'container': container}
            if o_opts is not None:
                object_options = deepcopy(options)
                object_options.update(o_opts)
            else:
                object_options = options
            if hasattr(s, 'read'):
                file_future = self.thread_manager.object_uu_pool.submit(self._upload_object_job, container, s, o, object_options, results_queue=rq)
                details['file'] = s
                details['object'] = o
                file_jobs[file_future] = details
            elif s is not None:
                details['path'] = s
                details['object'] = o
                if isdir(s):
                    dir_future = self.thread_manager.object_uu_pool.submit(self._create_dir_marker_job, container, o, object_options, path=s)
                    file_jobs[dir_future] = details
                else:
                    try:
                        stat(s)
                        file_future = self.thread_manager.object_uu_pool.submit(self._upload_object_job, container, s, o, object_options, results_queue=rq)
                        file_jobs[file_future] = details
                    except OSError as err:
                        traceback, err_time = report_traceback()
                        logger.exception(err)
                        res = {'action': 'upload_object', 'container': container, 'object': o, 'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time, 'path': s}
                        rq.put(res)
            else:
                details['file'] = None
                details['object'] = o
                if object_options['dir_marker']:
                    dir_future = self.thread_manager.object_uu_pool.submit(self._create_dir_marker_job, container, o, object_options)
                    file_jobs[dir_future] = details
                else:
                    file_future = self.thread_manager.object_uu_pool.submit(self._upload_object_job, container, StringIO(), o, object_options)
                    file_jobs[file_future] = details
        Thread(target=self._watch_futures, args=(file_jobs, rq)).start()
        res = get_from_queue(rq)
        cancelled = False
        while res is not None:
            yield res
            if not res['success']:
                if not cancelled and options['fail_fast']:
                    cancelled = True
                    for f in file_jobs:
                        f.cancel()
            res = get_from_queue(rq)

    @staticmethod
    def _make_upload_objects(objects, pseudo_folder=''):
        upload_objects = []
        for o in objects:
            if isinstance(o, str):
                obj = SwiftUploadObject(o, urljoin(pseudo_folder, o.lstrip('/')))
                upload_objects.append(obj)
            elif isinstance(o, SwiftUploadObject):
                o.object_name = urljoin(pseudo_folder, o.object_name)
                upload_objects.append(o)
            else:
                raise SwiftError('The upload operation takes only strings or SwiftUploadObjects as input', obj=o)
        return upload_objects

    @staticmethod
    def _create_container_job(conn, container, headers=None, policy_source=None):
        """
        Create a container using the given connection

        :param conn: The swift connection used for requests.
        :param container: The container name to create.
        :param headers: An optional dict of headers for the
                        put_container request.
        :param policy_source: An optional name of a container whose policy we
                              should duplicate.
        :return: A dict containing the results of the operation.
        """
        res = {'action': 'create_container', 'container': container, 'headers': headers}
        create_response = {}
        try:
            if policy_source is not None:
                _meta = conn.head_container(policy_source)
                if 'x-storage-policy' in _meta:
                    policy_header = {POLICY: _meta.get('x-storage-policy')}
                    if headers is None:
                        headers = policy_header
                    else:
                        headers.update(policy_header)
            conn.put_container(container, headers, response_dict=create_response)
            res.update({'success': True, 'response_dict': create_response})
        except Exception as err:
            traceback, err_time = report_traceback()
            logger.exception(err)
            res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time, 'response_dict': create_response})
        return res

    @staticmethod
    def _create_dir_marker_job(conn, container, obj, options, path=None):
        res = {'action': 'create_dir_marker', 'container': container, 'object': obj, 'path': path}
        results_dict = {}
        if obj.startswith('./') or obj.startswith('.\\'):
            obj = obj[2:]
        if obj.startswith('/'):
            obj = obj[1:]
        if path is not None:
            put_headers = {'x-object-meta-mtime': '%f' % getmtime(path)}
        else:
            put_headers = {'x-object-meta-mtime': '%f' % round(time())}
        res['headers'] = put_headers
        if options['changed']:
            try:
                headers = conn.head_object(container, obj)
                ct = headers.get('content-type', '').split(';', 1)[0]
                cl = int(headers.get('content-length'))
                et = headers.get('etag')
                mt = headers.get('x-object-meta-mtime')
                if ct in KNOWN_DIR_MARKERS and cl == 0 and (et == EMPTY_ETAG) and (mt == put_headers['x-object-meta-mtime']):
                    res['success'] = True
                    return res
            except ClientException as err:
                if err.http_status != 404:
                    traceback, err_time = report_traceback()
                    logger.exception(err)
                    res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time})
                    return res
        try:
            conn.put_object(container, obj, '', content_length=0, content_type=KNOWN_DIR_MARKERS[0], headers=put_headers, response_dict=results_dict)
            res.update({'success': True, 'response_dict': results_dict})
            return res
        except Exception as err:
            traceback, err_time = report_traceback()
            logger.exception(err)
            res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time, 'response_dict': results_dict})
            return res

    @staticmethod
    def _upload_segment_job(conn, path, container, segment_name, segment_start, segment_size, segment_index, obj_name, options, results_queue=None):
        results_dict = {}
        if options['segment_container']:
            segment_container = options['segment_container']
        else:
            segment_container = container + '_segments'
        res = {'action': 'upload_segment', 'for_container': container, 'for_object': obj_name, 'segment_index': segment_index, 'segment_size': segment_size, 'segment_location': '/%s/%s' % (segment_container, segment_name), 'log_line': '%s segment %s' % (obj_name, segment_index)}
        fp = None
        try:
            fp = open(path, 'rb', DISK_BUFFER)
            fp.seek(segment_start)
            contents = LengthWrapper(fp, segment_size, md5=options['checksum'])
            etag = conn.put_object(segment_container, segment_name, contents, content_length=segment_size, content_type='application/swiftclient-segment', response_dict=results_dict)
            if options['checksum'] and etag and (etag != contents.get_md5sum()):
                raise SwiftError('Segment {0}: upload verification failed: md5 mismatch, local {1} != remote {2} (remote segment has not been removed)'.format(segment_index, contents.get_md5sum(), etag))
            res.update({'success': True, 'response_dict': results_dict, 'segment_etag': etag, 'attempts': conn.attempts})
            if results_queue is not None:
                results_queue.put(res)
            return res
        except Exception as err:
            traceback, err_time = report_traceback()
            logger.exception(err)
            res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time, 'response_dict': results_dict, 'attempts': conn.attempts})
            if results_queue is not None:
                results_queue.put(res)
            return res
        finally:
            if fp is not None:
                fp.close()

    @staticmethod
    def _put_object(conn, container, name, content, headers=None, md5=None):
        """
        Upload object into a given container and verify the resulting ETag, if
        the md5 optional parameter is passed.

        :param conn: The Swift connection to use for uploads.
        :param container: The container to put the object into.
        :param name: The name of the object.
        :param content: Object content.
        :param headers: Headers (optional) to associate with the object.
        :param md5: MD5 sum of the content. If passed in, will be used to
                    verify the returned ETag.

        :returns: A dictionary as the response from calling put_object.
                  The keys are:
                    - status
                    - reason
                    - headers
                  On error, the dictionary contains the following keys:
                    - success (with value False)
                    - error - the encountered exception (object)
                    - error_timestamp
                    - response_dict - results from the put_object call, as
                      documented above
                    - attempts - number of attempts made
        """
        if headers is None:
            headers = {}
        else:
            headers = dict(headers)
        if md5 is not None:
            headers['etag'] = md5
        results = {}
        try:
            etag = conn.put_object(container, name, content, content_length=len(content), headers=headers, response_dict=results)
            if md5 is not None and etag != md5:
                raise SwiftError('Upload verification failed for {0}: md5 mismatch {1} != {2}'.format(name, md5, etag))
            results['success'] = True
        except Exception as err:
            traceback, err_time = report_traceback()
            logger.exception(err)
            return {'success': False, 'error': err, 'error_timestamp': err_time, 'response_dict': results, 'attempts': conn.attempts, 'traceback': traceback}
        return results

    @staticmethod
    def _upload_stream_segment(conn, container, object_name, segment_container, segment_name, segment_size, segment_index, headers, fd):
        """
        Upload a segment from a stream, buffering it in memory first. The
        resulting object is placed either as a segment in the segment
        container, or if it is smaller than a single segment, as the given
        object name.

        :param conn: Swift Connection to use.
        :param container: Container in which the object would be placed.
        :param object_name: Name of the final object (used in case the stream
                            is smaller than the segment_size)
        :param segment_container: Container to hold the object segments.
        :param segment_name: The name of the segment.
        :param segment_size: Minimum segment size.
        :param segment_index: The segment index.
        :param headers: Headers to attach to the segment/object.
        :param fd: File-like handle for the content. Must implement read().

        :returns: Dictionary, containing the following keys:
                    - complete -- whether the stream is exhausted
                    - segment_size - the actual size of the segment (may be
                                     smaller than the passed in segment_size)
                    - segment_location - path to the segment
                    - segment_index - index of the segment
                    - segment_etag - the ETag for the segment
        """
        buf = []
        dgst = md5()
        bytes_read = 0
        while bytes_read < segment_size:
            data = fd.read(segment_size - bytes_read)
            if not data:
                break
            bytes_read += len(data)
            dgst.update(data)
            buf.append(data)
        buf = b''.join(buf)
        segment_hash = dgst.hexdigest()
        if not buf and segment_index > 0:
            return {'complete': True, 'segment_size': 0, 'segment_index': None, 'segment_etag': None, 'segment_location': None, 'success': True}
        if segment_index == 0 and len(buf) < segment_size:
            ret = SwiftService._put_object(conn, container, object_name, buf, headers, segment_hash)
            ret['segment_location'] = '/%s/%s' % (container, object_name)
        else:
            ret = SwiftService._put_object(conn, segment_container, segment_name, buf, headers, segment_hash)
            ret['segment_location'] = '/%s/%s' % (segment_container, segment_name)
        ret.update(dict(complete=len(buf) < segment_size, segment_size=len(buf), segment_index=segment_index, segment_etag=segment_hash, for_object=object_name))
        return ret

    def _get_chunk_data(self, conn, container, obj, headers, manifest=None):
        chunks = []
        if 'x-object-manifest' in headers:
            scontainer, sprefix = headers['x-object-manifest'].split('/', 1)
            for part in self.list(scontainer, {'prefix': sprefix}):
                if part['success']:
                    chunks.extend(part['listing'])
                else:
                    raise part['error']
        elif config_true_value(headers.get('x-static-large-object')):
            if manifest is None:
                headers, manifest = conn.get_object(container, obj, query_string='multipart-manifest=get')
            manifest = parse_api_response(headers, manifest)
            for chunk in manifest:
                if chunk.get('sub_slo'):
                    scont, sobj = chunk['name'].lstrip('/').split('/', 1)
                    chunks.extend(self._get_chunk_data(conn, scont, sobj, {'x-static-large-object': True}))
                else:
                    chunks.append(chunk)
        else:
            chunks.append({'hash': headers.get('etag').strip('"'), 'bytes': int(headers.get('content-length'))})
        return chunks

    def _is_identical(self, chunk_data, path):
        if path is None:
            return False
        try:
            fp = open(path, 'rb', DISK_BUFFER)
        except IOError:
            return False
        with fp:
            for chunk in chunk_data:
                to_read = chunk['bytes']
                md5sum = md5()
                while to_read:
                    data = fp.read(min(DISK_BUFFER, to_read))
                    if not data:
                        return False
                    md5sum.update(data)
                    to_read -= len(data)
                if md5sum.hexdigest() != chunk['hash']:
                    return False
            return not fp.read(1)

    @staticmethod
    def _upload_slo_manifest(conn, segment_results, container, obj, headers):
        """
        Upload an SLO manifest, given the results of uploading each segment, to
        the specified container.

        :param segment_results: List of response_dict structures, as populated
                                by _upload_segment_job. Specifically, each
                                entry must container the following keys:
                                - segment_location
                                - segment_etag
                                - segment_size
                                - segment_index
        :param container: The container to put the manifest into.
        :param obj: The name of the manifest object to use.
        :param headers: Optional set of headers to attach to the manifest.
        """
        if headers is None:
            headers = {}
        segment_results.sort(key=lambda di: di['segment_index'])
        manifest_data = json.dumps([{'path': d['segment_location'], 'etag': d['segment_etag'], 'size_bytes': d['segment_size']} for d in segment_results])
        response = {}
        conn.put_object(container, obj, manifest_data, headers=headers, query_string='multipart-manifest=put', response_dict=response)
        return response

    def _upload_object_job(self, conn, container, source, obj, options, results_queue=None):
        if obj.startswith('./') or obj.startswith('.\\'):
            obj = obj[2:]
        if obj.startswith('/'):
            obj = obj[1:]
        res = {'action': 'upload_object', 'container': container, 'object': obj}
        if hasattr(source, 'read'):
            stream = source
            path = None
        else:
            path = source
        res['path'] = path
        try:
            if path is not None:
                put_headers = {'x-object-meta-mtime': '%f' % getmtime(path)}
            else:
                put_headers = {'x-object-meta-mtime': '%f' % round(time())}
            res['headers'] = put_headers
            old_manifest = None
            old_slo_manifest_paths = []
            new_slo_manifest_paths = set()
            segment_size = int(0 if options['segment_size'] is None else options['segment_size'])
            if options['changed'] or options['skip_identical'] or (not options['leave_segments']):
                try:
                    headers = conn.head_object(container, obj)
                    is_slo = config_true_value(headers.get('x-static-large-object'))
                    if options['skip_identical'] or (is_slo and (not options['leave_segments'])):
                        chunk_data = self._get_chunk_data(conn, container, obj, headers)
                    if options['skip_identical'] and self._is_identical(chunk_data, path):
                        res.update({'success': True, 'status': 'skipped-identical'})
                        return res
                    cl = int(headers.get('content-length'))
                    mt = headers.get('x-object-meta-mtime')
                    if path is not None and options['changed'] and (cl == getsize(path)) and (mt == put_headers['x-object-meta-mtime']):
                        res.update({'success': True, 'status': 'skipped-changed'})
                        return res
                    if not options['leave_segments'] and (not headers.get('content-location')):
                        old_manifest = headers.get('x-object-manifest')
                        if is_slo:
                            old_slo_manifest_paths.extend((normalize_manifest_path(old_seg['name']) for old_seg in chunk_data))
                except ClientException as err:
                    if err.http_status != 404:
                        traceback, err_time = report_traceback()
                        logger.exception(err)
                        res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time})
                        return res
            put_headers.update(split_headers(options['meta'], 'X-Object-Meta-'))
            put_headers.update(split_headers(options['header'], ''))
            if path is not None and segment_size and (getsize(path) > segment_size):
                res['large_object'] = True
                seg_container = container + '_segments'
                if options['segment_container']:
                    seg_container = options['segment_container']
                full_size = getsize(path)
                segment_futures = []
                segment_pool = self.thread_manager.segment_pool
                segment = 0
                segment_start = 0
                while segment_start < full_size:
                    if segment_start + segment_size > full_size:
                        segment_size = full_size - segment_start
                    if options['use_slo']:
                        segment_name = '%s/slo/%s/%s/%s/%08d' % (obj, put_headers['x-object-meta-mtime'], full_size, options['segment_size'], segment)
                    else:
                        segment_name = '%s/%s/%s/%s/%08d' % (obj, put_headers['x-object-meta-mtime'], full_size, options['segment_size'], segment)
                    seg = segment_pool.submit(self._upload_segment_job, path, container, segment_name, segment_start, segment_size, segment, obj, options, results_queue=results_queue)
                    segment_futures.append(seg)
                    segment += 1
                    segment_start += segment_size
                segment_results = []
                errors = False
                exceptions = []
                for f in interruptable_as_completed(segment_futures):
                    try:
                        r = f.result()
                        if not r['success']:
                            errors = True
                        segment_results.append(r)
                    except Exception as err:
                        traceback, err_time = report_traceback()
                        logger.exception(err)
                        errors = True
                        exceptions.append((err, traceback, err_time))
                if errors:
                    err = ClientException('Aborting manifest creation because not all segments could be uploaded. %s/%s' % (container, obj))
                    res.update({'success': False, 'error': err, 'exceptions': exceptions, 'segment_results': segment_results})
                    return res
                res['segment_results'] = segment_results
                if options['use_slo']:
                    response = self._upload_slo_manifest(conn, segment_results, container, obj, put_headers)
                    res['manifest_response_dict'] = response
                    new_slo_manifest_paths.update((normalize_manifest_path(new_seg['segment_location']) for new_seg in segment_results))
                else:
                    new_object_manifest = '%s/%s/%s/%s/%s/' % (quote(seg_container.encode('utf8')), quote(obj.encode('utf8')), put_headers['x-object-meta-mtime'], full_size, options['segment_size'])
                    if old_manifest and old_manifest.rstrip('/') == new_object_manifest.rstrip('/'):
                        old_manifest = None
                    put_headers['x-object-manifest'] = new_object_manifest
                    mr = {}
                    conn.put_object(container, obj, '', content_length=0, headers=put_headers, response_dict=mr)
                    res['manifest_response_dict'] = mr
            elif options['use_slo'] and segment_size and (not path):
                segment = 0
                results = []
                while True:
                    segment_name = '%s/slo/%s/%s/%08d' % (obj, put_headers['x-object-meta-mtime'], segment_size, segment)
                    seg_container = container + '_segments'
                    if options['segment_container']:
                        seg_container = options['segment_container']
                    ret = self._upload_stream_segment(conn, container, obj, seg_container, segment_name, segment_size, segment, put_headers, stream)
                    if not ret['success']:
                        return ret
                    if ret['complete'] and segment == 0 or ret['segment_size'] > 0:
                        results.append(ret)
                    if results_queue is not None:
                        if ret['segment_location'] != '/%s/%s' % (container, obj) and ret['segment_size'] > 0:
                            results_queue.put(ret)
                    if ret['complete']:
                        break
                    segment += 1
                if results[0]['segment_location'] != '/%s/%s' % (container, obj):
                    response = self._upload_slo_manifest(conn, results, container, obj, put_headers)
                    res['manifest_response_dict'] = response
                    new_slo_manifest_paths.update((normalize_manifest_path(new_seg['segment_location']) for new_seg in results))
                    res['large_object'] = True
                else:
                    res['response_dict'] = ret
                    res['large_object'] = False
            else:
                res['large_object'] = False
                obr = {}
                fp = None
                try:
                    if path is not None:
                        content_length = getsize(path)
                        fp = open(path, 'rb', DISK_BUFFER)
                        contents = LengthWrapper(fp, content_length, md5=options['checksum'])
                    else:
                        content_length = None
                        contents = ReadableToIterable(stream, md5=options['checksum'])
                    etag = conn.put_object(container, obj, contents, content_length=content_length, headers=put_headers, response_dict=obr)
                    res['response_dict'] = obr
                    if options['checksum'] and etag and (etag != contents.get_md5sum()):
                        raise SwiftError('Object upload verification failed: md5 mismatch, local {0} != remote {1} (remote object has not been removed)'.format(contents.get_md5sum(), etag))
                finally:
                    if fp is not None:
                        fp.close()
            if old_manifest or old_slo_manifest_paths:
                drs = []
                delobjsmap = defaultdict(list)
                if old_manifest:
                    scontainer, sprefix = old_manifest.split('/', 1)
                    sprefix = sprefix.rstrip('/') + '/'
                    for part in self.list(scontainer, {'prefix': sprefix}):
                        if not part['success']:
                            raise part['error']
                        delobjsmap[scontainer].extend((seg['name'] for seg in part['listing']))
                if old_slo_manifest_paths:
                    for seg_to_delete in old_slo_manifest_paths:
                        if seg_to_delete in new_slo_manifest_paths:
                            continue
                        scont, sobj = seg_to_delete.split('/', 1)
                        delobjsmap[scont].append(sobj)
                del_segs = []
                for dscont, dsobjs in delobjsmap.items():
                    for dsobj in dsobjs:
                        del_seg = self.thread_manager.segment_pool.submit(self._delete_segment, dscont, dsobj, results_queue=results_queue)
                        del_segs.append(del_seg)
                for del_seg in interruptable_as_completed(del_segs):
                    drs.append(del_seg.result())
                res['segment_delete_results'] = drs
            res.update({'success': True, 'status': 'uploaded', 'attempts': conn.attempts})
            return res
        except OSError as err:
            traceback, err_time = report_traceback()
            logger.exception(err)
            if err.errno == ENOENT:
                error = SwiftError('Local file %r not found' % path, exc=err)
            else:
                error = err
            res.update({'success': False, 'error': error, 'traceback': traceback, 'error_timestamp': err_time})
        except Exception as err:
            traceback, err_time = report_traceback()
            logger.exception(err)
            res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time})
        return res

    def delete(self, container=None, objects=None, options=None):
        """
        Delete operations on an account, optional container and optional list
        of objects.

        :param container: The container to delete or delete from.
        :param objects: A list of object names (strings) or SwiftDeleteObject
                        instances containing an object name, and an
                        options dict (can be None) to override the options for
                        that individual delete operation::

                            [
                                'object_name',
                                SwiftDeleteObject('object_name',
                                                  options={...}),
                                ...
                            ]

                        The options dict is described below.
        :param options: A dictionary containing options to override the global
                        options specified during the service object creation::

                            {
                                'yes_all': False,
                                'leave_segments': False,
                                'version_id': None,
                                'prefix': None,
                                'versions': False,
                                'header': [],
                            }

        :returns: A generator for returning the results of the delete
                  operations. Each result yielded from the generator is either
                  a 'delete_container', 'delete_object', 'delete_segment', or
                  'bulk_delete' dictionary containing the results of an
                  individual delete operation.

        :raises ClientException:
        :raises SwiftError:
        """
        if options is not None:
            options = dict(self._options, **options)
        else:
            options = self._options
        if container is not None:
            if objects is not None:
                delete_objects = self._make_delete_objects(objects)
                if options['prefix']:
                    delete_objects = [obj for obj in delete_objects if obj.object_name.startswith(options['prefix'])]
                rq = Queue()
                obj_dels = {}
                bulk_page_size = self._bulk_delete_page_size(delete_objects)
                if bulk_page_size > 1:
                    page_at_a_time = n_at_a_time(delete_objects, bulk_page_size)
                    for page_slice in page_at_a_time:
                        for obj_slice in n_groups(page_slice, self._options['object_dd_threads']):
                            object_names = [obj.object_name for obj in obj_slice]
                            self._bulk_delete(container, object_names, options, obj_dels)
                else:
                    self._per_item_delete(container, delete_objects, options, obj_dels, rq)
                Thread(target=self._watch_futures, args=(obj_dels, rq)).start()
                res = get_from_queue(rq)
                while res is not None:
                    yield res
                    if options['fail_fast'] and (not res['success']):
                        for d in obj_dels.keys():
                            d.cancel()
                    res = get_from_queue(rq)
            else:
                for res in self._delete_container(container, options):
                    yield res
        else:
            if objects:
                raise SwiftError('Objects specified without container')
            if options['prefix']:
                raise SwiftError('Prefix specified without container')
            if options['yes_all']:
                cancelled = False
                containers = []
                for part in self.list():
                    if part['success']:
                        containers.extend((c['name'] for c in part['listing']))
                    else:
                        raise part['error']
                for con in containers:
                    if cancelled:
                        break
                    else:
                        for res in self._delete_container(con, options=options):
                            yield res
                            if not cancelled and options['fail_fast'] and (not res['success']):
                                cancelled = True

    def _bulk_delete_page_size(self, objects):
        """
        Given the iterable 'objects', will return how many items should be
        deleted at a time.

        :param objects: An iterable that supports 'len()'
        :returns: The bulk delete page size (i.e. the max number of
                  objects that can be bulk deleted at once, as reported by
                  the cluster). If bulk delete is disabled, return 1
        """
        if len(objects) <= 2 * self._options['object_dd_threads']:
            return 1
        if any((obj.options for obj in objects if isinstance(obj, SwiftDeleteObject))):
            return 1
        try:
            cap_result = self.capabilities()
            if not cap_result['success']:
                return 1
        except ClientException:
            return 1
        swift_info = cap_result['capabilities']
        if 'bulk_delete' in swift_info:
            return swift_info['bulk_delete'].get('max_deletes_per_request', 10000)
        else:
            return 1

    def _per_item_delete(self, container, objects, options, rdict, rq):
        for delete_obj in objects:
            obj = delete_obj.object_name
            obj_options = dict(options, **delete_obj.options or {})
            obj_del = self.thread_manager.object_dd_pool.submit(self._delete_object, container, obj, obj_options, results_queue=rq)
            obj_details = {'container': container, 'object': obj}
            rdict[obj_del] = obj_details

    @staticmethod
    def _delete_segment(conn, container, obj, results_queue=None):
        results_dict = {}
        try:
            res = {'success': True}
            conn.delete_object(container, obj, response_dict=results_dict)
        except Exception as err:
            if not isinstance(err, ClientException) or err.http_status != 404:
                traceback, err_time = report_traceback()
                logger.exception(err)
                res = {'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time}
        res.update({'action': 'delete_segment', 'container': container, 'object': obj, 'attempts': conn.attempts, 'response_dict': results_dict})
        if results_queue is not None:
            results_queue.put(res)
        return res

    @staticmethod
    def _make_delete_objects(objects):
        delete_objects = []
        for o in objects:
            if isinstance(o, str):
                obj = SwiftDeleteObject(o)
                delete_objects.append(obj)
            elif isinstance(o, SwiftDeleteObject):
                delete_objects.append(o)
            else:
                raise SwiftError('The delete operation takes only strings or SwiftDeleteObjects as input', obj=o)
        return delete_objects

    def _delete_object(self, conn, container, obj, options, results_queue=None):
        _headers = {}
        _headers = split_headers(options.get('header', []))
        res = {'action': 'delete_object', 'container': container, 'object': obj}
        try:
            old_manifest = None
            query_params = {}
            if not options['leave_segments']:
                try:
                    headers = conn.head_object(container, obj, headers=_headers, query_string='symlink=get')
                    old_manifest = headers.get('x-object-manifest')
                    if config_true_value(headers.get('x-static-large-object')):
                        query_params['multipart-manifest'] = 'delete'
                except ClientException as err:
                    if err.http_status != 404:
                        raise
            if options.get('version_id') is not None:
                query_params['version-id'] = options['version_id']
            query_string = '&'.join(('%s=%s' % (k, v) for k, v in sorted(query_params.items())))
            results_dict = {}
            conn.delete_object(container, obj, headers=_headers, query_string=query_string, response_dict=results_dict)
            if old_manifest:
                dlo_segments_deleted = True
                segment_pool = self.thread_manager.segment_pool
                s_container, s_prefix = old_manifest.split('/', 1)
                s_prefix = s_prefix.rstrip('/') + '/'
                del_segs = []
                for part in self.list(container=s_container, options={'prefix': s_prefix}):
                    if part['success']:
                        seg_list = [o['name'] for o in part['listing']]
                    else:
                        raise part['error']
                    for seg in seg_list:
                        del_seg = segment_pool.submit(self._delete_segment, s_container, seg, results_queue=results_queue)
                        del_segs.append(del_seg)
                for del_seg in interruptable_as_completed(del_segs):
                    del_res = del_seg.result()
                    if not del_res['success']:
                        dlo_segments_deleted = False
                res['dlo_segments_deleted'] = dlo_segments_deleted
            res.update({'success': True, 'response_dict': results_dict, 'attempts': conn.attempts})
        except Exception as err:
            traceback, err_time = report_traceback()
            logger.exception(err)
            res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time})
            return res
        return res

    @staticmethod
    def _delete_empty_container(conn, container, options):
        results_dict = {}
        _headers = {}
        _headers = split_headers(options.get('header', []))
        try:
            conn.delete_container(container, headers=_headers, response_dict=results_dict)
            res = {'success': True}
        except Exception as err:
            traceback, err_time = report_traceback()
            logger.exception(err)
            res = {'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time}
        res.update({'action': 'delete_container', 'container': container, 'object': None, 'attempts': conn.attempts, 'response_dict': results_dict})
        return res

    def _delete_container(self, container, options):
        try:
            for part in self.list(container=container, options=options):
                if not part['success']:
                    raise part['error']
                delete_objects = []
                for item in part['listing']:
                    delete_opts = {}
                    if options.get('versions', False) and 'version_id' in item:
                        delete_opts['version_id'] = item['version_id']
                    delete_obj = SwiftDeleteObject(item['name'], delete_opts)
                    delete_objects.append(delete_obj)
                for res in self.delete(container=container, objects=delete_objects, options=options):
                    yield res
            if options['prefix']:
                return
            con_del = self.thread_manager.container_pool.submit(self._delete_empty_container, container, options)
            con_del_res = get_future_result(con_del)
        except Exception as err:
            traceback, err_time = report_traceback()
            logger.exception(err)
            con_del_res = {'action': 'delete_container', 'container': container, 'object': None, 'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time}
        yield con_del_res

    def _bulk_delete(self, container, objects, options, rdict):
        if objects:
            bulk_del = self.thread_manager.object_dd_pool.submit(self._bulkdelete, container, objects, options)
            bulk_details = {'container': container, 'objects': objects}
            rdict[bulk_del] = bulk_details

    @staticmethod
    def _bulkdelete(conn, container, objects, options):
        results_dict = {}
        try:
            headers = {'Accept': 'application/json', 'Content-Type': 'text/plain'}
            res = {'container': container, 'objects': objects}
            objects = [quote(('/%s/%s' % (container, obj)).encode('utf-8')) for obj in objects]
            headers, body = conn.post_account(headers=headers, query_string='bulk-delete', data=b''.join((obj.encode('utf-8') + b'\n' for obj in objects)), response_dict=results_dict)
            if body:
                res.update({'success': True, 'result': parse_api_response(headers, body)})
            else:
                res.update({'success': False, 'error': SwiftError('No content received on account POST. Is the bulk operations middleware enabled?')})
        except Exception as e:
            traceback, err_time = report_traceback()
            logger.exception(e)
            res.update({'success': False, 'error': e, 'traceback': traceback})
        res.update({'action': 'bulk_delete', 'attempts': conn.attempts, 'response_dict': results_dict})
        return res

    def copy(self, container, objects, options=None):
        """
        Copy operations on a list of objects in a container. Destination
        containers will be created.

        :param container: The container from which to copy the objects.
        :param objects: A list of object names (strings) or SwiftCopyObject
                        instances containing an object name and an
                        options dict (can be None) to override the options for
                        that individual copy operation::

                            [
                                'object_name',
                                SwiftCopyObject(
                                    'object_name',
                                     options={
                                        'destination': '/container/object',
                                        'fresh_metadata': False,
                                        ...
                                        }),
                                ...
                            ]

                        The options dict is described below.
        :param options: A dictionary containing options to override the global
                        options specified during the service object creation.
                        These options are applied to all copy operations
                        performed by this call, unless overridden on a per
                        object basis.
                        The options "destination" and "fresh_metadata" do
                        not need to be set, in this case objects will be
                        copied onto themselves and metadata will not be
                        refreshed.
                        The option "destination" can also be specified in the
                        format '/container', in which case objects without an
                        explicit destination will be copied to the destination
                        /container/original_object_name. Combinations of
                        multiple objects and a destination in the format
                        '/container/object' is invalid. Possible options are
                        given below::

                            {
                                'meta': [],
                                'header': [],
                                'destination': '/container/object',
                                'fresh_metadata': False,
                            }

        :returns: A generator returning the results of copying the given list
                  of objects.

        :raises SwiftError:
        """
        if options is not None:
            options = dict(self._options, **options)
        else:
            options = self._options
        containers = set((next((p for p in obj.destination.split('/') if p)) for obj in objects if isinstance(obj, SwiftCopyObject) and obj.destination))
        if options.get('destination'):
            destination_split = options['destination'].split('/')
            if destination_split[0]:
                raise SwiftError('destination must be in format /cont[/obj]')
            _str_objs = [o for o in objects if not isinstance(o, SwiftCopyObject)]
            if len(destination_split) > 2 and len(_str_objs) > 1:
                raise SwiftError('Combination of multiple objects and destination including object is invalid')
            if destination_split[-1] == '':
                raise SwiftError('destination can not end in a slash')
            containers.add(destination_split[1])
        policy_header = {}
        _header = split_headers(options['header'])
        if POLICY in _header:
            policy_header[POLICY] = _header[POLICY]
        create_containers = [self.thread_manager.container_pool.submit(self._create_container_job, cont, headers=policy_header) for cont in containers]
        for r in interruptable_as_completed(create_containers):
            res = r.result()
            yield res
        copy_futures = []
        copy_objects = self._make_copy_objects(objects, options)
        for copy_object in copy_objects:
            obj = copy_object.object_name
            obj_options = copy_object.options
            destination = copy_object.destination
            fresh_metadata = copy_object.fresh_metadata
            headers = split_headers(options['meta'], 'X-Object-Meta-')
            headers.update(split_headers(options['header'], ''))
            if obj_options is not None:
                if 'meta' in obj_options:
                    headers.update(split_headers(obj_options['meta'], 'X-Object-Meta-'))
                if 'header' in obj_options:
                    headers.update(split_headers(obj_options['header'], ''))
            copy = self.thread_manager.object_uu_pool.submit(self._copy_object_job, container, obj, destination, headers, fresh_metadata)
            copy_futures.append(copy)
        for r in interruptable_as_completed(copy_futures):
            res = r.result()
            yield res

    @staticmethod
    def _make_copy_objects(objects, options):
        copy_objects = []
        for o in objects:
            if isinstance(o, str):
                obj = SwiftCopyObject(o, options)
                copy_objects.append(obj)
            elif isinstance(o, SwiftCopyObject):
                copy_objects.append(o)
            else:
                raise SwiftError('The copy operation takes only strings or SwiftCopyObjects as input', obj=o)
        return copy_objects

    @staticmethod
    def _copy_object_job(conn, container, obj, destination, headers, fresh_metadata):
        response_dict = {}
        res = {'success': True, 'action': 'copy_object', 'container': container, 'object': obj, 'destination': destination, 'headers': headers, 'fresh_metadata': fresh_metadata, 'response_dict': response_dict}
        try:
            conn.copy_object(container, obj, destination=destination, headers=headers, fresh_metadata=fresh_metadata, response_dict=response_dict)
        except Exception as err:
            traceback, err_time = report_traceback()
            logger.exception(err)
            res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time})
        return res

    def capabilities(self, url=None, refresh_cache=False):
        """
        List the cluster capabilities.

        :param url: Proxy URL of the cluster to retrieve capabilities.

        :returns: A dictionary containing the capabilities of the cluster.

        :raises ClientException:
        """
        if not refresh_cache and url in self.capabilities_cache:
            return self.capabilities_cache[url]
        res = {'action': 'capabilities', 'timestamp': time()}
        cap = self.thread_manager.container_pool.submit(self._get_capabilities, url)
        capabilities = get_future_result(cap)
        res.update({'success': True, 'capabilities': capabilities})
        if url is not None:
            res.update({'url': url})
        self.capabilities_cache[url] = res
        return res

    @staticmethod
    def _get_capabilities(conn, url):
        return conn.get_capabilities(url)

    @staticmethod
    def _watch_futures(futures, result_queue):
        """
        Watches a dict of futures and pushes their results onto the given
        queue. We use this to wait for a set of futures which may create
        futures of their own to wait for, whilst also allowing us to
        immediately return the results of those sub-jobs.

        When all futures have completed, None is pushed to the queue

        If the future is cancelled, we use the dict to return details about
        the cancellation.
        """
        futures_only = list(futures.keys())
        for f in interruptable_as_completed(futures_only):
            try:
                r = f.result()
                if r is not None:
                    result_queue.put(r)
            except CancelledError:
                details = futures[f]
                res = details
                res['status'] = 'cancelled'
                result_queue.put(res)
            except Exception as err:
                traceback, err_time = report_traceback()
                logger.exception(err)
                details = futures[f]
                res = details
                res.update({'success': False, 'error': err, 'traceback': traceback, 'error_timestamp': err_time})
                result_queue.put(res)
        result_queue.put(None)