import requests
from bs4 import BeautifulSoup
import os
import time

# Constants
BASE_URL = "https://www.freepik.com/search?format=search&last_filter=type&last_value=vector&query=kids+coloring&selection=1&type=vector"
DOWNLOAD_DIR = "coloring_images"
REQUEST_DELAY = 2  # seconds

# Create download directory if it doesn't exist
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)


def get_image_links(page_url):
    response = requests.get(page_url)
    soup = BeautifulSoup(response.content, "html.parser")
    image_tags = soup.find_all("img", class_="showcase__image")
    image_links = [img["src"] for img in image_tags if "src" in img.attrs]
    return image_links


def download_image(url, folder, image_num):
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(folder, f"image_{image_num}.jpg"), "wb") as file:
            file.write(response.content)
        print(f"Downloaded {url}")
    else:
        print(f"Failed to download {url}")


def main():
    page_num = 1
    image_num = 1
    while True:
        page_url = f"{BASE_URL}&page={page_num}"
        image_links = get_image_links(page_url)
        if not image_links:
            break  # No more images found, exit the loop
        for link in image_links:
            download_image(link, DOWNLOAD_DIR, image_num)
            image_num += 1
            time.sleep(REQUEST_DELAY)
        page_num += 1


if __name__ == "__main__":
    main()
