import requests
import aiohttp
import asyncio
from dataclasses import dataclass
from lazyops.lazyclasses import lazyclass
from typing import List, Dict, Any, Optional
class LazySession:

    def __init__(self, header=None):
        self.header = header
        self.sess = requests.session()
        if header:
            self.sess.headers.update(header)

    def fetch(self, decode_json=True, *args, **kwargs):
        res = self.sess.request(*args, **kwargs)
        if decode_json:
            return res.json()
        return res

    async def async_batch(self, url, batch_params, decode_json=True, *args, **kwargs):
        tasks = []
        async with aiohttp.ClientSession(headers=self.header) as sess:
            for batch in batch_params:
                tasks.append(asyncio.ensure_future(async_req(*args, sess=sess, url=url, **batch, **kwargs)))
            all_tasks = await asyncio.gather(*tasks)
            return [task for task in all_tasks]

    async def async_fetch_urls(self, urls, decode_json=True, *args, **kwargs):
        tasks = []
        async with aiohttp.ClientSession(headers=self.header) as sess:
            for url in urls:
                tasks.append(asyncio.ensure_future(async_req(*args, sess=sess, url=url, with_url=True, **kwargs)))
            all_tasks = await asyncio.gather(*tasks)
            return [task for task in all_tasks]

    async def async_fetch(self, decode_json=True, *args, **kwargs):
        async with aiohttp.ClientSession(headers=self.header) as sess:
            async with sess.request(*args, **kwargs) as resp:
                return await resp.json() if decode_json else await resp

    def __call__(self, *args, **kwargs):
        return self.fetch(*args, **kwargs)

    def __exit__(self, _):
        self.sess.close()