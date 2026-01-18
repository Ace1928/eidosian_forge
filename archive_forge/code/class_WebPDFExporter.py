import asyncio
import concurrent.futures
import os
import subprocess
import sys
import tempfile
from importlib import util as importlib_util
from traitlets import Bool, default
from .html import HTMLExporter
class WebPDFExporter(HTMLExporter):
    """Writer designed to write to PDF files.

    This inherits from :class:`HTMLExporter`. It creates the HTML using the
    template machinery, and then run playwright to create a pdf.
    """
    export_from_notebook = 'PDF via HTML'
    allow_chromium_download = Bool(False, help='Whether to allow downloading Chromium if no suitable version is found on the system.').tag(config=True)
    paginate = Bool(True, help='\n        Split generated notebook into multiple pages.\n\n        If False, a PDF with one long page will be generated.\n\n        Set to True to match behavior of LaTeX based PDF generator\n        ').tag(config=True)

    @default('file_extension')
    def _file_extension_default(self):
        return '.html'

    @default('template_name')
    def _template_name_default(self):
        return 'webpdf'
    disable_sandbox = Bool(False, help='\n        Disable chromium security sandbox when converting to PDF.\n\n        WARNING: This could cause arbitrary code execution in specific circumstances,\n        where JS in your notebook can execute serverside code! Please use with\n        caution.\n\n        ``https://github.com/puppeteer/puppeteer/blob/main@%7B2020-12-14T17:22:24Z%7D/docs/troubleshooting.md#setting-up-chrome-linux-sandbox``\n        has more information.\n\n        This is required for webpdf to work inside most container environments.\n        ').tag(config=True)

    def run_playwright(self, html):
        """Run playwright."""

        async def main(temp_file):
            """Run main playwright script."""
            args = ['--no-sandbox'] if self.disable_sandbox else []
            try:
                from playwright.async_api import async_playwright
            except ModuleNotFoundError as e:
                msg = 'Playwright is not installed to support Web PDF conversion. Please install `nbconvert[webpdf]` to enable.'
                raise RuntimeError(msg) from e
            if self.allow_chromium_download:
                cmd = [sys.executable, '-m', 'playwright', 'install', 'chromium']
                subprocess.check_call(cmd)
            playwright = await async_playwright().start()
            chromium = playwright.chromium
            try:
                browser = await chromium.launch(handle_sigint=False, handle_sigterm=False, handle_sighup=False, args=args)
            except Exception as e:
                msg = "No suitable chromium executable found on the system. Please use '--allow-chromium-download' to allow downloading one,or install it using `playwright install chromium`."
                await playwright.stop()
                raise RuntimeError(msg) from e
            page = await browser.new_page()
            await page.emulate_media(media='print')
            await page.wait_for_timeout(100)
            await page.goto(f'file://{temp_file.name}', wait_until='networkidle')
            await page.wait_for_timeout(100)
            pdf_params = {'print_background': True}
            if not self.paginate:
                dimensions = await page.evaluate('() => {\n                    const rect = document.body.getBoundingClientRect();\n                    return {\n                    width: Math.ceil(rect.width) + 1,\n                    height: Math.ceil(rect.height) + 1,\n                    }\n                }')
                width = dimensions['width']
                height = dimensions['height']
                pdf_params.update({'width': min(width, 200 * 72), 'height': min(height, 200 * 72)})
            pdf_data = await page.pdf(**pdf_params)
            await browser.close()
            await playwright.stop()
            return pdf_data
        pool = concurrent.futures.ThreadPoolExecutor()
        temp_file = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
        with temp_file:
            temp_file.write(html.encode('utf-8'))
        try:

            def run_coroutine(coro):
                """Run an internal coroutine."""
                loop = asyncio.ProactorEventLoop() if IS_WINDOWS else asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            pdf_data = pool.submit(run_coroutine, main(temp_file)).result()
        finally:
            os.unlink(temp_file.name)
        return pdf_data

    def from_notebook_node(self, nb, resources=None, **kw):
        """Convert from a notebook node."""
        html, resources = super().from_notebook_node(nb, resources=resources, **kw)
        self.log.info('Building PDF')
        pdf_data = self.run_playwright(html)
        self.log.info('PDF successfully created')
        resources['output_extension'] = '.pdf'
        return (pdf_data, resources)